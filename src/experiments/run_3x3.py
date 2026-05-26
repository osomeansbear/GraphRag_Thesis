"""
run_3x3.py — Thesis evaluation experiment runner.

Evaluation design:
  3 modes x 28 questions (questions_multihop_focused.json).

    Mode               Role
    VECTOR_RETRIEVAL   Dense vector baseline — no graph traversal, no reranker
    GRAPH_RETRIEVAL    PPR over L2+L3 graph + query-entity seeding + CrossEncoder filter
    GRAPH_ITERATIVE    GRAPH_RETRIEVAL + one reflection cycle (IRCoT-style gap fill)

  Metrics: BERTScore F1 (local, no API) + ROUGE-L + Recall@3 + Hit@3 + confidence + latency.
  LLM: SingleGroqLLM pinned to llama-3.1-8b-instant (retry-only, no model substitution).

Features:
  - Resume capability: skips cells whose (id, mode) already in output CSV
    with a non-empty answer (status=success).
  - status field per cell: success | no_context | generation_error.
  - Inter-call sleep: small constant delay between generation calls to avoid
    triggering RPM limits.
  - Failure counter: aborts the run if >50% of the last 50 cells failed.

Usage:
  python src/experiments/run_3x3.py --mode VECTOR_RETRIEVAL --output outputs/eval_vec.csv
  python src/experiments/run_3x3.py --mode GRAPH_RETRIEVAL --output outputs/eval_graph.csv
  python src/experiments/run_3x3.py --mode GRAPH_ITERATIVE --output outputs/eval_iter.csv
  python src/experiments/run_3x3.py --score-only --output outputs/eval_vec.csv
"""

import argparse
import json
import os
import sys
import time
from collections import deque
import re

import pandas as pd
from llama_index.core.llms import ChatMessage
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.single_groq_llm import SingleGroqLLM
from src.engine.retriever import ThesisRetriever

_THINK_RE = re.compile(r'<\s*think\s*>.*?<\s*/\s*think\s*>', re.IGNORECASE | re.DOTALL)

def _strip_think(text: str) -> str:
    return re.sub(r'\n{3,}', '\n\n', _THINK_RE.sub('', text)).strip()

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
MODES = ["VECTOR_RETRIEVAL", "GRAPH_RETRIEVAL", "GRAPH_ITERATIVE"]

DEFAULT_QUESTIONS = "data/test_set/questions_multihop_focused.json"
OUTPUT_DIR = "outputs"

BERTSCORE_MODEL = "bert-base-uncased"

# Only k=3 is reported — matches the retriever's top-3 output window.
HITK_CUTOFFS = [3]

# ---------------------------------------------------------------------------
# Generation policy
# ---------------------------------------------------------------------------
INTER_CALL_SLEEP_S = 2.0
FAILURE_WINDOW = 50
FAILURE_ABORT_RATIO = 0.5

GENERATION_SYSTEM_PROMPT = """You are an Academic Regulations Assistant.
Answer the question based solely on the CONTEXT provided below.
RULES:
- If the context contains "REASONING CHAIN", prioritize that information.
- Cite specific articles when possible using [Article N] or [Clause N, Article M].
- If the information is not in the context, say: "I could not find this in the regulations."
- Keep answers concise and factual.

CONTEXT:
{context}
"""


def _to_key(pair) -> tuple:
    """Coerce a [art, cl] pair (from JSON) or tuple to a hashable (int, int)."""
    if pair is None or len(pair) < 2:
        return (None, None)
    a, c = pair[0], pair[1]
    return (int(a) if a is not None else None,
            int(c) if c is not None else None)


def compute_hitk(retrieved_keys: list, source_keys: list) -> dict:
    """
    Compute Hit@k (binary: at least one source clause in top-k) and
    Recall@k (fraction of source clauses retrieved in top-k).
    """
    out = {}
    if not source_keys:
        for k in HITK_CUTOFFS:
            out[f"hit@{k}"] = None
            out[f"recall@{k}"] = None
        return out

    src_set = {_to_key(p) for p in source_keys}
    src_set.discard((None, None))
    if not src_set:
        for k in HITK_CUTOFFS:
            out[f"hit@{k}"] = None
            out[f"recall@{k}"] = None
        return out

    retrieved = [_to_key(p) for p in (retrieved_keys or [])]

    for k in HITK_CUTOFFS:
        topk = set(retrieved[:k])
        intersect = topk & src_set
        out[f"hit@{k}"] = 1 if intersect else 0
        out[f"recall@{k}"] = round(len(intersect) / len(src_set), 3)
    return out


# ---------------------------------------------------------------------------
# BERTScore evaluation
# ---------------------------------------------------------------------------

def run_scoring(records: list, scores_path: str) -> list:
    """
    Compute BERTScore F1 and ROUGE-L for all status=success rows.
    Both run locally — no API calls. Results written to scores_path.
    """
    from bert_score import score as _bertscore
    from rouge_score import rouge_scorer as _rouge_scorer

    scorable = [r for r in records if r.get("status") == "success"]
    if not scorable:
        print("No scorable rows for scoring.")
        return records

    hyps = [str(r.get("answer", "")) for r in scorable]
    refs = [str(r.get("ground_truth", "")) for r in scorable]

    print(f"\nComputing BERTScore ({BERTSCORE_MODEL}) for {len(scorable)} rows...")
    _, _, F1 = _bertscore(hyps, refs, model_type=BERTSCORE_MODEL, verbose=False, device="cpu")
    for record, f1 in zip(scorable, F1.tolist()):
        record["bertscore_f1"] = round(f1, 4)

    print(f"Computing ROUGE-L for {len(scorable)} rows...")
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    for record, hyp, ref in zip(scorable, hyps, refs):
        record["rougeL"] = round(scorer.score(ref, hyp)["rougeL"].fmeasure, 4)

    _save_scores(records, scores_path)
    print(f"Scoring complete — saved to {scores_path}")
    return records


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def load_existing_results(path: str) -> set:
    """Return set of (question_id, mode) keys already present with status=success."""
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        if "status" in df.columns:
            df = df[df["status"] == "success"]
        return set(zip(df["question_id"], df["mode"]))
    except Exception:
        return set()


def generate_answer(llm, context_str: str, query: str) -> tuple[str, str]:
    """Single LLM call: context + query → (answer, status)."""
    messages = [
        ChatMessage(
            role="system",
            content=GENERATION_SYSTEM_PROMPT.format(context=context_str),
        ),
        ChatMessage(role="user", content=query),
    ]
    try:
        response = llm.chat(messages)
        return _strip_think(str(response)), "success"
    except Exception as e:
        return f"[Generation error: {e}]", "generation_error"


def run_generation(
    question_file: str,
    selected_modes: list,
    sleep_seconds: float,
    output_path: str,
    ids_filter: list = None,
):
    """Step 1: produce one (question, mode) row per cell with a status field."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {question_file}")

    if ids_filter:
        questions = [q for q in questions if q["id"] in ids_filter]
        print(f"Filtered to {len(questions)} questions by --ids: {ids_filter}")

    done = load_existing_results(output_path)
    if done:
        print(f"Resuming — {len(done)} (question, mode) cells already succeeded; skipping those.\n")

    retriever = ThesisRetriever()
    llm = SingleGroqLLM(model="llama-3.1-8b-instant")
    print(f"[Generation LLM] Groq ({llm._sg_model_name})")
    retriever._reflect_llm = llm

    total_cells = len(questions) * len(selected_modes)
    print(f"Matrix: {len(selected_modes)} modes x {len(questions)} questions = {total_cells} cells\n")

    new_records = []
    failure_window: deque = deque(maxlen=FAILURE_WINDOW)

    pbar = tqdm(total=total_cells, desc="Cells")
    for mode in selected_modes:
        for q in questions:
            pbar.set_description(f"{mode}/{q['id']}")

            if (q["id"], mode) in done:
                pbar.update(1)
                continue

            t0 = time.perf_counter()
            context_str, confidence, _, retrieved_keys = retriever.search(
                q["query"], mode=mode,
            )

            if not context_str:
                answer = "No relevant information found in the regulations."
                status = "no_context"
            else:
                answer, status = generate_answer(llm, context_str, q["query"])

            latency_ms = (time.perf_counter() - t0) * 1000

            hitk = compute_hitk(retrieved_keys, q.get("source_clause_id"))

            gap_logged = getattr(retriever, "last_gap", None) if mode == "GRAPH_ITERATIVE" else None

            record = {
                "question_id":    q["id"],
                "question_type":  q.get("type", ""),
                "mode":           mode,
                "query":          q["query"],
                "ground_truth":   q.get("ground_truth", ""),
                "answer":         answer,
                "confidence":     confidence,
                "latency_ms":     round(latency_ms, 1),
                "status":         status,
                "reflection_gap": gap_logged,
                "bertscore_f1":   None,
                "rougeL":         None,
                **hitk,
            }
            new_records.append(record)

            failure_window.append(0 if status == "success" else 1)
            if len(failure_window) == FAILURE_WINDOW:
                fail_ratio = sum(failure_window) / FAILURE_WINDOW
                if fail_ratio > FAILURE_ABORT_RATIO:
                    pbar.close()
                    print(
                        f"\nFailure ratio {fail_ratio:.0%} in last {FAILURE_WINDOW} cells "
                        "exceeds 50%. Aborting."
                    )
                    _save_scores(new_records, output_path)
                    raise SystemExit(1)

            pbar.update(1)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    pbar.close()

    if not new_records:
        print("No new cells generated. All done.")
        return []

    print(f"\nGenerated {len(new_records)} new answers.")
    return new_records


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_scores(records: list, path: str):
    """Upsert records into a scores CSV at path."""
    if not records:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    new_df = pd.DataFrame(records)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        keys = set(zip(new_df["question_id"], new_df["mode"]))
        existing_keys = list(zip(existing["question_id"], existing["mode"]))
        keep = [k not in keys for k in existing_keys]
        existing = existing[keep]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False, encoding="utf-8")


def _print_summary(scores_path: str = None):
    path = scores_path
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "success"]

    focus_cols = [c for c in ["bertscore_f1", "rougeL", "recall@3", "hit@3"] if c in df.columns and df[c].notna().any()]
    if not focus_cols:
        print("No scores available yet.")
        return

    print("\n" + "=" * 70)
    print("TABLE 1 -- Overall 2-mode comparison (mean)")
    print("=" * 70)
    means = df.groupby("mode")[focus_cols].mean().round(3)
    print(means.to_string())

    if "VECTOR_RETRIEVAL" in means.index and "GRAPH_RETRIEVAL" in means.index:
        delta = (means.loc["GRAPH_RETRIEVAL"] - means.loc["VECTOR_RETRIEVAL"]).round(3)
        print(f"\n  Delta (GRAPH - VECTOR): {delta.to_dict()}")

    if "question_type" in df.columns and df["question_type"].notna().any():
        print("\n" + "=" * 70)
        print("TABLE 2 -- Per-question-type breakdown")
        print("=" * 70)
        table2 = df.groupby(["mode", "question_type"])[focus_cols].mean().round(3)
        print(table2.to_string())

    print("=" * 70)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    question_file: str,
    selected_modes: list,
    score_only: bool,
    sleep_seconds: float,
    scores_path: str = None,
    ids_filter: list = None,
):
    effective_scores_path = scores_path

    if score_only:
        if not os.path.exists(effective_scores_path):
            print(f"--score-only requires existing {effective_scores_path}; none found.")
            return
        df = pd.read_csv(effective_scores_path)
        records = df.to_dict(orient="records")
        run_scoring(records, effective_scores_path)
        _print_summary(effective_scores_path)
        return

    new_records = run_generation(question_file, selected_modes, sleep_seconds, effective_scores_path, ids_filter)
    if not new_records:
        return

    _save_scores(new_records, effective_scores_path)
    print(f"Generated answers saved -> {effective_scores_path}")

    run_scoring(new_records, effective_scores_path)
    _print_summary(effective_scores_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Tier 1 experiment matrix")
    parser.add_argument(
        "--mode",
        choices=MODES + ["ALL"],
        default="ALL",
        help="Run a single retrieval mode (default: ALL). Choices: VECTOR_RETRIEVAL, GRAPH_RETRIEVAL, GRAPH_ITERATIVE.",
    )
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS,
        help=f"Path to question JSON file (default: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        dest="score_only",
        help="Compute BERTScore on existing answers in the CSV; no generation.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=INTER_CALL_SLEEP_S,
        help=f"Inter-call sleep in seconds (default {INTER_CALL_SLEEP_S})",
    )
    parser.add_argument(
        "--output",
        default=None,
        dest="scores_path",
        help=(
            "Write scores to this CSV. "
            "Use a per-mode path for parallel runs. "
            "Example: --output outputs/eval_vec.csv"
        ),
    )
    parser.add_argument(
        "--ids",
        default=None,
        dest="ids_filter",
        help="Comma-separated question IDs to run (e.g. Q08,Q23).",
    )
    args = parser.parse_args()

    selected_modes = MODES if args.mode == "ALL" else [args.mode]
    ids_filter = [x.strip() for x in args.ids_filter.split(",")] if args.ids_filter else None
    run(
        question_file=args.questions,
        selected_modes=selected_modes,
        score_only=args.score_only,
        sleep_seconds=args.sleep,
        scores_path=args.scores_path,
        ids_filter=ids_filter,
    )
