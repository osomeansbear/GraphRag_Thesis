"""
investigate_recall.py — Diagnose the Recall@3 paradox.

Problem: GRAPH_RETRIEVAL wins ROUGE-L (+0.042, p=0.017*) but loses Recall@3 (-0.062).
This means GRAPH generates answers lexically closer to ground truth, yet retrieves
fewer of the labelled source clauses. Two hypotheses:
  A) GRAPH retrieves semantically-related unlabelled clauses that happen to contain
     vocabulary the ground-truth answer also uses (PPR is finding useful context
     outside the gold-label set).
  B) GRAPH drops source clauses and replaces them with noise the LLM ignores at
     generation time (the ROUGE gain comes from something else).

This script replays retrieval for all questions (no generation, no LLM calls),
captures retrieved clause IDs, and produces:
  - outputs/recall_audit.csv   full per-cell breakdown
  - console: paradox table + extra-clause text for manual inspection

Usage:
  python scripts/investigate_recall.py
  python scripts/investigate_recall.py --modes VECTOR_RETRIEVAL GRAPH_RETRIEVAL
  python scripts/investigate_recall.py --ids Q09 Q11 Q19
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.retriever import ThesisRetriever

QUESTIONS_FILE = "data/test_set/questions_multihop_focused.json"
SCORE_FILES = {
    "VECTOR_RETRIEVAL": "outputs/ragas_vec.csv",
    "GRAPH_RETRIEVAL":  "outputs/ragas_graph_07.csv",
}
OUTPUT_CSV = "outputs/recall_audit.csv"
DEFAULT_MODES = ["VECTOR_RETRIEVAL", "GRAPH_RETRIEVAL"]


def _to_key(pair) -> tuple:
    if pair is None or len(pair) < 2:
        return (None, None)
    a, c = pair[0], pair[1]
    return (int(a) if a is not None else None,
            int(c) if c is not None else None)


def load_scores() -> pd.DataFrame:
    """Load existing BERTScore / ROUGE / Recall@3 from saved CSVs."""
    frames = []
    for _, path in SCORE_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[df["status"] == "success"]
            frames.append(df[["question_id", "mode", "bertscore_f1", "rougeL", "recall@3", "hit@3"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_clause_text(driver, article_num, clause_num) -> str:
    """Fetch raw clause text from Neo4j for a given (article, clause) key."""
    if article_num is None or clause_num is None:
        return "(unknown clause)"
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (c:Clause {article_num: $a, clause_num: $cl}) "
                "RETURN c.content_raw AS text LIMIT 1",
                a=int(article_num), cl=int(clause_num),
            )
            rec = result.single()
            if rec:
                text = rec["text"] or ""
                return text[:200] + ("..." if len(text) > 200 else "")
    except Exception:
        pass
    return "(fetch failed)"


def run_audit(modes: list, ids_filter: list = None):
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if ids_filter:
        questions = [q for q in questions if q["id"] in ids_filter]

    scores_df = load_scores()
    retriever = ThesisRetriever()

    records = []
    total = len(questions) * len(modes)
    done = 0

    for mode in modes:
        for q in questions:
            done += 1
            qid = q["id"]
            qtype = q.get("type", "")
            source_keys = {_to_key(p) for p in (q.get("source_clause_id") or [])}
            source_keys.discard((None, None))

            print(f"[{done}/{total}] {mode} / {qid} ({qtype})", end=" ... ", flush=True)

            _, _, _, retrieved_keys_raw = retriever.search(
                q["query"], mode=mode, graph_level=3, anchor_weight=0.7,
            )

            retrieved_keys = [_to_key(k) for k in (retrieved_keys_raw or [])]
            retrieved_set = set(retrieved_keys)

            missed = source_keys - retrieved_set
            extra = retrieved_set - source_keys

            recall_recomputed = (
                round(len(source_keys & retrieved_set) / len(source_keys), 3)
                if source_keys else None
            )
            hit_recomputed = 1 if (source_keys & retrieved_set) else 0

            # Pull saved scores for this (qid, mode) if available
            saved = pd.DataFrame()
            if not scores_df.empty:
                saved = scores_df[
                    (scores_df["question_id"] == qid) & (scores_df["mode"] == mode)
                ]

            bertscore = float(saved["bertscore_f1"].iloc[0]) if not saved.empty else None
            rougeL    = float(saved["rougeL"].iloc[0])       if not saved.empty else None
            recall_saved = float(saved["recall@3"].iloc[0])  if not saved.empty else None

            records.append({
                "question_id":       qid,
                "question_type":     qtype,
                "mode":              mode,
                "source_keys":       str(sorted(source_keys)),
                "retrieved_keys":    str(retrieved_keys),
                "missed_keys":       str(sorted(missed)),
                "extra_keys":        str(sorted(extra)),
                "recall@3_recomputed": recall_recomputed,
                "hit@3_recomputed":  hit_recomputed,
                "recall@3_saved":    recall_saved,
                "bertscore_f1":      bertscore,
                "rougeL":            rougeL,
                "n_source":          len(source_keys),
                "n_retrieved":       len(retrieved_keys),
                "n_missed":          len(missed),
                "n_extra":           len(extra),
            })
            print(f"recall={recall_recomputed} hit={hit_recomputed} missed={sorted(missed)} extra={sorted(extra)}")

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nFull audit saved -> {OUTPUT_CSV}\n")

    _print_summary(df, retriever.driver)


def _print_summary(df: pd.DataFrame, driver):
    print("=" * 72)
    print("RECALL AUDIT SUMMARY")
    print("=" * 72)

    # Overall recall by mode
    print("\n--- Mean Recall@3 (recomputed from live retrieval) ---")
    print(df.groupby("mode")["recall@3_recomputed"].mean().round(3).to_string())

    # Missed-key frequency table
    print("\n--- Source clauses missed by GRAPH_RETRIEVAL ---")
    graph_rows = df[df["mode"] == "GRAPH_RETRIEVAL"]
    missed_counts: dict = {}
    for _, row in graph_rows.iterrows():
        if row["n_missed"] > 0:
            for key_str in eval(row["missed_keys"]):
                missed_counts[key_str] = missed_counts.get(key_str, 0) + 1
    for k, v in sorted(missed_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: missed {v}x")

    # Paradox cases: GRAPH BERTScore > VECTOR AND GRAPH Recall@3 < VECTOR
    print("\n--- Paradox cases (GRAPH better BERTScore but worse Recall@3) ---")
    if df["bertscore_f1"].isna().all():
        print("  No BERTScore saved yet for Q25-Q28. Run evaluation first.")
        scored = df[df["bertscore_f1"].notna()].copy()
    else:
        scored = df[df["bertscore_f1"].notna()].copy()

    if not scored.empty:
        vec = scored[scored["mode"] == "VECTOR_RETRIEVAL"].set_index("question_id")
        gph = scored[scored["mode"] == "GRAPH_RETRIEVAL"].set_index("question_id")
        common = vec.index.intersection(gph.index)
        paradox_ids = [
            qid for qid in common
            if (gph.loc[qid, "bertscore_f1"] > vec.loc[qid, "bertscore_f1"])
            and (gph.loc[qid, "recall@3_recomputed"] < vec.loc[qid, "recall@3_recomputed"])
        ]

        if paradox_ids:
            print(f"  Found {len(paradox_ids)} paradox question(s): {paradox_ids}\n")
            for qid in paradox_ids:
                v_row = vec.loc[qid]
                g_row = gph.loc[qid]
                print(f"  {qid} ({g_row['question_type']})")
                print(f"    BERTScore: VEC={v_row['bertscore_f1']:.3f}  GRAPH={g_row['bertscore_f1']:.3f}  d={g_row['bertscore_f1']-v_row['bertscore_f1']:+.3f}")
                print(f"    Recall@3:  VEC={v_row['recall@3_recomputed']:.3f}  GRAPH={g_row['recall@3_recomputed']:.3f}  d={g_row['recall@3_recomputed']-v_row['recall@3_recomputed']:+.3f}")
                print(f"    Source keys:  {v_row['source_keys']}")
                print(f"    VEC retrieved:{v_row['retrieved_keys']}")
                print(f"    GRAPH retrieved:{g_row['retrieved_keys']}")
                print(f"    GRAPH missed: {g_row['missed_keys']}")
                extra_keys = eval(g_row["extra_keys"]) if g_row["extra_keys"] != "set()" else []
                if extra_keys:
                    print(f"    GRAPH extra clauses (fetched text):")
                    for (art, cl) in extra_keys:
                        text = fetch_clause_text(driver, art, cl)
                        print(f"      Art {art} Cl {cl}: {text}")
                print()
        else:
            print("  No paradox cases found in scored questions.")
    else:
        print("  No scored data available — run evaluation first then rerun this script.")

    # Questions where BOTH modes fail (Recall@3 = 0.0 or all missed)
    print("\n--- Questions where GRAPH retrieves 0 source clauses ---")
    zero_recall = df[
        (df["mode"] == "GRAPH_RETRIEVAL") & (df["recall@3_recomputed"] == 0.0)
    ][["question_id", "question_type", "source_keys", "retrieved_keys"]]
    if zero_recall.empty:
        print("  None — all questions have at least partial source coverage.")
    else:
        print(zero_recall.to_string(index=False))

    print("\n" + "=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recall@3 audit — replay retrieval, diagnose paradox")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=["VECTOR_RETRIEVAL", "GRAPH_RETRIEVAL"],
        help="Retrieval modes to audit (default: both)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Specific question IDs to audit (default: all 28)",
    )
    args = parser.parse_args()
    run_audit(modes=args.modes, ids_filter=args.ids)
