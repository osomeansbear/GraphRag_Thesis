"""
analyse_results.py -- Chapter 6 analysis for the 3-mode evaluation.

Covers:
  Table 1 -- Overall per-mode means (BERTScore F1, ROUGE-L, Recall@3, latency)
  Table 2 -- Per-mode x question-type means (CROSSREF vs MULTIHOP)
  Table 3 -- Paired Wilcoxon H1: GRAPH_RETRIEVAL > VECTOR_RETRIEVAL
  Table 4 -- H1 stratified by question type
  Table 5 -- Retrieval metrics (Hit@3, Recall@3)
  Table 6 -- Narrative summary for Chapter 6

Usage:
    python scripts/analyse_results.py
    python scripts/analyse_results.py --results outputs/ragas_vec.csv outputs/ragas_graph_07.csv outputs/ragas_iter.csv
    python scripts/analyse_results.py --output outputs/analysis.md
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_QUESTIONS = os.path.join("data", "test_set", "questions_multihop_focused.json")
DEFAULT_RESULTS = [
    os.path.join("outputs", "eval_vec.csv"),
    os.path.join("outputs", "eval_graph.csv"),
    os.path.join("outputs", "eval_iter.csv"),
]
DEFAULT_OUTPUT  = os.path.join("outputs", "analysis.md")

MODES_ORDERED  = ["VECTOR_RETRIEVAL", "GRAPH_RETRIEVAL", "GRAPH_ITERATIVE"]
TYPES_ORDERED  = ["CROSSREF", "MULTIHOP"]
SCORE_COLS     = ["bertscore_f1", "rougeL"]
RETRIEVAL_COLS = ["hit@3", "recall@3", "precision@3", "complete@3"]

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(paths: list[str], questions_path: str = DEFAULT_QUESTIONS) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not os.path.exists(path):
            print(f"WARNING: results file not found, skipping: {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        print("ERROR: no results files found.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    n_missing = df["bertscore_f1"].isna().sum() if "bertscore_f1" in df.columns else len(df)
    if n_missing:
        print(f"WARNING: {n_missing} rows missing bertscore_f1 — run --score-only first.")
    success = df[df["status"] == "success"].copy()
    print(f"Loaded {len(df)} total rows across {len(frames)} file(s); {len(success)} status=success for analysis.")

    # Enrich with Precision@3 and Complete@3 from questions JSON.
    # precision@3 = recall@3 * n_source_clauses / 3
    # complete@3  = 1 if ALL source clauses were found in top-3 (recall@3 == 1.0)
    if os.path.exists(questions_path) and "recall@3" in success.columns:
        with open(questions_path, encoding="utf-8") as f:
            qs = {q["id"]: len(q.get("source_clause_id") or []) for q in __import__("json").load(f)}
        success["_n_source"] = success["question_id"].map(qs).fillna(1)
        success["precision@3"] = (success["recall@3"] * success["_n_source"] / 3).round(3)
        success["complete@3"]  = (success["recall@3"] == 1.0).astype(int)
        success = success.drop(columns=["_n_source"])
    else:
        print("WARNING: questions file not found — precision@3 and complete@3 unavailable.")

    return success


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    return f"{v:.3f}"


def pval_str(p: float) -> str:
    if p != p:
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def sig_marker(p: float) -> str:
    if p != p:
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def wilcoxon_pair(a: pd.Series, b: pd.Series,
                  alternative: str = "greater") -> tuple[float, float, int]:
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    n = len(paired)
    if n < 5:
        return float("nan"), float("nan"), n
    stat, p = stats.wilcoxon(paired["a"], paired["b"], alternative=alternative)
    return float(stat), float(p), n


def ttest_pair(a: pd.Series, b: pd.Series,
               alternative: str = "greater") -> tuple[float, float, int]:
    """Paired t-test (parametric complement to Wilcoxon).
    alternative='greater' tests H1: a > b (i.e. GRAPH > VECTOR).
    """
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    n = len(paired)
    if n < 5:
        return float("nan"), float("nan"), n
    t, p_two = stats.ttest_rel(paired["a"], paired["b"])
    if alternative == "greater":
        p = p_two / 2 if t >= 0 else 1 - p_two / 2
    elif alternative == "less":
        p = p_two / 2 if t <= 0 else 1 - p_two / 2
    else:
        p = p_two
    return float(t), float(p), n


def md_table(headers: list[str], rows: list[list]) -> str:
    col_w = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep  = "| " + " | ".join("-" * w for w in col_w) + " |"
    head = "| " + " | ".join(str(h).ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(col_w[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 1 -- Overall per-mode means
# ---------------------------------------------------------------------------

def section_overall_means(df: pd.DataFrame) -> str:
    lines = ["## Table 1 -- Overall per-mode scores\n"]
    rows = []
    for mode in MODES_ORDERED:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        n    = len(sub)
        bs_m = sub["bertscore_f1"].mean() if "bertscore_f1" in sub.columns else float("nan")
        bs_s = sub["bertscore_f1"].std()  if "bertscore_f1" in sub.columns else float("nan")
        rl_m = sub["rougeL"].mean()       if "rougeL" in sub.columns else float("nan")
        r3_m = sub["recall@3"].mean()     if "recall@3" in sub.columns else float("nan")
        lat  = sub["latency_ms"].mean()   if "latency_ms" in sub.columns else float("nan")
        rows.append([
            mode, n,
            f"{fmt(bs_m)} ({fmt(bs_s)})",
            f"{fmt(rl_m)}",
            f"{fmt(r3_m)}",
            f"{lat:.0f}" if lat == lat else "N/A",
        ])
    headers = ["Mode", "N", "BERTScore F1 mean (sd)", "ROUGE-L mean", "Recall@3 mean", "Latency ms mean"]
    lines.append(md_table(headers, rows))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2 -- Per-mode x question-type means
# ---------------------------------------------------------------------------

def section_per_type_means(df: pd.DataFrame) -> str:
    lines = ["## Table 2 -- Per-mode x question-type scores\n"]
    for metric_label, col in [("BERTScore F1", "bertscore_f1"),
                               ("ROUGE-L", "rougeL"),
                               ("Recall@3", "recall@3"),
                               ("Complete@3", "complete@3")]:
        if col not in df.columns:
            continue
        lines.append(f"\n### {metric_label}\n")
        pivot = df.pivot_table(values=col, index="question_type",
                               columns="mode", aggfunc="mean")
        cols       = [m for m in MODES_ORDERED if m in pivot.columns]
        rows_order = [t for t in TYPES_ORDERED if t in pivot.index]
        pivot      = pivot.reindex(index=rows_order, columns=cols)
        headers    = ["Type"] + list(pivot.columns)
        rows = []
        for qtype in pivot.index:
            row = [qtype] + [fmt(pivot.at[qtype, c]) for c in pivot.columns]
            n_vec   = len(df[(df["mode"] == "VECTOR_RETRIEVAL") & (df["question_type"] == qtype)])
            n_graph = len(df[(df["mode"] == "GRAPH_RETRIEVAL")  & (df["question_type"] == qtype)])
            row.append(f"N={n_vec}/{n_graph}")
            rows.append(row)
        overall = ["**Overall**"] + [fmt(df[df["mode"] == c][col].mean()) for c in pivot.columns] + [""]
        rows.append(overall)
        lines.append(md_table(headers + ["N (VEC/GR)"], rows))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3 -- H1 Wilcoxon: GRAPH_RETRIEVAL > VECTOR_RETRIEVAL
# ---------------------------------------------------------------------------

def section_wilcoxon_main(df: pd.DataFrame) -> str:
    lines = [
        "## Table 3 -- Paired Wilcoxon signed-rank test (proposed > VECTOR baseline)\n",
        f"alpha = {ALPHA}, one-tailed.\n",
    ]
    vec_df = df[df["mode"] == "VECTOR_RETRIEVAL"].set_index("question_id")
    challengers = [m for m in MODES_ORDERED if m != "VECTOR_RETRIEVAL"]

    for challenger in challengers:
        chal_df = df[df["mode"] == challenger].set_index("question_id")
        common  = vec_df.index.intersection(chal_df.index)
        lines.append(f"\n### {challenger} vs VECTOR_RETRIEVAL\n")
        headers = ["Metric", "N pairs", "VECTOR mean", "Challenger mean",
                   "Delta", "Wilcoxon p", "t-test p", "Sig (W)", "Sig (t)"]
        rows = []
        for col, label in [("bertscore_f1", "BERTScore F1"),
                           ("rougeL",       "ROUGE-L"),
                           ("recall@3",     "Recall@3"),
                           ("complete@3",   "Complete@3")]:
            if col not in vec_df.columns or col not in chal_df.columns:
                continue
            v = vec_df.loc[common, col]
            g = chal_df.loc[common, col]
            _, pw, n = wilcoxon_pair(g, v, "greater")
            _, pt, _ = ttest_pair(g, v, "greater")
            rows.append([label, n, fmt(v.mean()), fmt(g.mean()),
                         fmt(g.mean() - v.mean()), pval_str(pw), pval_str(pt),
                         sig_marker(pw), sig_marker(pt)])
        if rows:
            lines.append(md_table(headers, rows))

    lines.append("\n_*p<0.05  **p<0.01  ***p<0.001  ns=not significant_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4 -- H1 stratified by question type
# ---------------------------------------------------------------------------

def section_wilcoxon_stratified(df: pd.DataFrame) -> str:
    lines = ["## Table 4 -- Wilcoxon stratified by question type (BERTScore F1)\n"]
    if "bertscore_f1" not in df.columns:
        return "## Table 4 -- Wilcoxon stratified\n\n_bertscore_f1 column not found._"
    vec_df     = df[df["mode"] == "VECTOR_RETRIEVAL"].set_index("question_id")
    challengers = [m for m in MODES_ORDERED if m != "VECTOR_RETRIEVAL"]
    headers = ["Challenger", "Type", "N pairs", "VECTOR mean", "Challenger mean",
               "Delta", "Wilcoxon p", "t-test p", "Sig (W)", "Sig (t)"]
    rows = []
    for challenger in challengers:
        chal_df = df[df["mode"] == challenger].set_index("question_id")
        for qtype in TYPES_ORDERED:
            v_sub = vec_df[vec_df["question_type"] == qtype]
            g_sub = chal_df[chal_df["question_type"] == qtype]
            common = v_sub.index.intersection(g_sub.index)
            if len(common) < 5:
                rows.append([challenger, qtype, len(common),
                             "N/A", "N/A", "N/A", "N/A (N<5)", "N/A (N<5)", "N/A", "N/A"])
                continue
            v = v_sub.loc[common, "bertscore_f1"]
            g = g_sub.loc[common, "bertscore_f1"]
            _, pw, n = wilcoxon_pair(g, v, "greater")
            _, pt, _ = ttest_pair(g, v, "greater")
            rows.append([challenger, qtype, n, fmt(v.mean()), fmt(g.mean()),
                         fmt(g.mean() - v.mean()), pval_str(pw), pval_str(pt),
                         sig_marker(pw), sig_marker(pt)])

    lines.append(md_table(headers, rows))
    lines.append("\n_*p<0.05  **p<0.01  ***p<0.001  ns=not significant_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 5 -- Retrieval metrics
# ---------------------------------------------------------------------------

def section_retrieval(df: pd.DataFrame) -> str:
    lines = ["## Table 5 -- Retrieval metrics (Hit@3, Recall@3)\n"]
    avail = [c for c in RETRIEVAL_COLS if c in df.columns]
    if not avail:
        return "## Table 5 -- Retrieval metrics\n\n_No retrieval columns found._"
    rows = []
    for mode in MODES_ORDERED:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        rows.append([mode, len(sub)] + [fmt(sub[c].mean()) for c in avail])
    for qtype in TYPES_ORDERED:
        for mode in MODES_ORDERED:
            sub = df[(df["mode"] == mode) & (df["question_type"] == qtype)]
            if sub.empty:
                continue
            rows.append([f"{mode}/{qtype}", len(sub)] + [fmt(sub[c].mean()) for c in avail])
    headers = ["Mode", "N"] + avail
    lines.append(md_table(headers, rows))
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Table 6 -- Narrative summary
# ---------------------------------------------------------------------------

def section_summary(df: pd.DataFrame) -> str:
    lines = ["## Table 7 -- Narrative summary for Chapter 6\n"]

    if "bertscore_f1" not in df.columns:
        return "## Table 7 -- Narrative summary\n\n_bertscore_f1 column not found._"

    mode_means = df.groupby("mode")["bertscore_f1"].mean()
    best_mode  = mode_means.idxmax() if not mode_means.empty else "N/A"
    worst_mode = mode_means.idxmin() if not mode_means.empty else "N/A"
    lines.append(f"- Best mode (BERTScore F1): **{best_mode}** "
                 f"({fmt(mode_means.get(best_mode, float('nan')))})")
    lines.append(f"- Worst mode: **{worst_mode}** "
                 f"({fmt(mode_means.get(worst_mode, float('nan')))})")

    vec_m = mode_means.get("VECTOR_RETRIEVAL", float("nan"))
    for challenger in [m for m in MODES_ORDERED if m != "VECTOR_RETRIEVAL"]:
        chal_m = mode_means.get(challenger, float("nan"))
        if vec_m == vec_m and chal_m == chal_m:
            lines.append(f"- Delta {challenger} vs VECTOR (BERTScore F1): {fmt(chal_m - vec_m)}")

    vec_df = df[df["mode"] == "VECTOR_RETRIEVAL"].set_index("question_id")
    for challenger in [m for m in MODES_ORDERED if m != "VECTOR_RETRIEVAL"]:
        chal_df = df[df["mode"] == challenger].set_index("question_id")
        common  = vec_df.index.intersection(chal_df.index)
        if len(common) < 5:
            continue
        g_bs = chal_df.loc[common, "bertscore_f1"]
        v_bs = vec_df.loc[common, "bertscore_f1"]
        _, pw, n = wilcoxon_pair(g_bs, v_bs, "greater")
        _, pt, _ = ttest_pair(g_bs, v_bs, "greater")
        result_w = "SIGNIFICANT" if (pw == pw and pw < ALPHA) else "not significant"
        result_t = "SIGNIFICANT" if (pt == pt and pt < ALPHA) else "not significant"
        lines.append(f"- {challenger} overall (N={n}): Wilcoxon p={pval_str(pw)} {sig_marker(pw)} -- {result_w}")
        lines.append(f"- {challenger} overall (N={n}):  t-test  p={pval_str(pt)} {sig_marker(pt)} -- {result_t}")
        for qtype in TYPES_ORDERED:
            v_sub = vec_df[vec_df["question_type"] == qtype]
            g_sub = chal_df[chal_df["question_type"] == qtype]
            comm  = v_sub.index.intersection(g_sub.index)
            if len(comm) < 5:
                lines.append(f"  - {challenger}/{qtype}: N={len(comm)} (insufficient)")
                continue
            g_bs2 = g_sub.loc[comm, "bertscore_f1"]
            v_bs2 = v_sub.loc[comm, "bertscore_f1"]
            _, pw2, n2 = wilcoxon_pair(g_bs2, v_bs2, "greater")
            _, pt2, _  = ttest_pair(g_bs2, v_bs2, "greater")
            d = g_bs2.mean() - v_bs2.mean()
            lines.append(
                f"  - {challenger}/{qtype} (N={n2}): delta={fmt(d)}, "
                f"Wilcoxon p={pval_str(pw2)} {sig_marker(pw2)}, "
                f"t-test p={pval_str(pt2)} {sig_marker(pt2)}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_tables_csv(df_main: pd.DataFrame, tables_dir: str) -> None:
    os.makedirs(tables_dir, exist_ok=True)

    # Table 1 -- overall means
    rows = []
    for mode in MODES_ORDERED:
        sub = df_main[df_main["mode"] == mode]
        if sub.empty:
            continue
        row = {"mode": mode, "n": len(sub)}
        for col in SCORE_COLS + RETRIEVAL_COLS:
            if col in sub.columns:
                row[col] = round(sub[col].mean(), 4)
        if "latency_ms" in sub.columns:
            row["latency_ms_mean"] = round(sub["latency_ms"].mean(), 1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "overall_means.csv"), index=False)

    # Table 2 -- per-type means (one row per mode x type x metric)
    rows = []
    for mode in MODES_ORDERED:
        for qtype in TYPES_ORDERED:
            sub = df_main[(df_main["mode"] == mode) & (df_main["question_type"] == qtype)]
            if sub.empty:
                continue
            row = {"mode": mode, "question_type": qtype, "n": len(sub)}
            for col in SCORE_COLS + RETRIEVAL_COLS:
                if col in sub.columns:
                    row[col] = round(sub[col].mean(), 4)
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "per_type_means.csv"), index=False)

    # Table 3 -- Wilcoxon main (all challengers vs VECTOR)
    vec_df = df_main[df_main["mode"] == "VECTOR_RETRIEVAL"].set_index("question_id")
    rows = []
    for challenger in [m for m in MODES_ORDERED if m != "VECTOR_RETRIEVAL"]:
        chal_df = df_main[df_main["mode"] == challenger].set_index("question_id")
        common  = vec_df.index.intersection(chal_df.index)
        for col, label in [("bertscore_f1", "BERTScore F1"), ("rougeL", "ROUGE-L"),
                           ("recall@3", "Recall@3"), ("complete@3", "Complete@3")]:
            if col not in vec_df.columns or col not in chal_df.columns:
                continue
            v, g = vec_df.loc[common, col], chal_df.loc[common, col]
            _, pw, n = wilcoxon_pair(g, v, "greater")
            _, pt, _ = ttest_pair(g, v, "greater")
            rows.append({"challenger": challenger, "metric": label, "n_pairs": n,
                         "vector_mean": round(v.mean(), 4), "challenger_mean": round(g.mean(), 4),
                         "delta": round(g.mean() - v.mean(), 4),
                         "wilcoxon_p": round(pw, 4) if pw == pw else None,
                         "ttest_p": round(pt, 4) if pt == pt else None,
                         "sig_wilcoxon": sig_marker(pw), "sig_ttest": sig_marker(pt)})
    pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "wilcoxon.csv"), index=False)

    # Table 4 -- Wilcoxon stratified (all challengers vs VECTOR, by question type)
    rows = []
    if "bertscore_f1" in vec_df.columns:
        for challenger in [m for m in MODES_ORDERED if m != "VECTOR_RETRIEVAL"]:
            chal_df = df_main[df_main["mode"] == challenger].set_index("question_id")
            for qtype in TYPES_ORDERED:
                v_sub = vec_df[vec_df["question_type"] == qtype]
                g_sub = chal_df[chal_df["question_type"] == qtype]
                comm  = v_sub.index.intersection(g_sub.index)
                if len(comm) < 5:
                    rows.append({"challenger": challenger, "question_type": qtype,
                                 "n_pairs": len(comm), "note": "N<5"})
                    continue
                v, g = v_sub.loc[comm, "bertscore_f1"], g_sub.loc[comm, "bertscore_f1"]
                _, pw, n = wilcoxon_pair(g, v, "greater")
                _, pt, _ = ttest_pair(g, v, "greater")
                rows.append({"challenger": challenger, "question_type": qtype, "n_pairs": n,
                             "vector_mean": round(v.mean(), 4), "challenger_mean": round(g.mean(), 4),
                             "delta": round(g.mean() - v.mean(), 4),
                             "wilcoxon_p": round(pw, 4) if pw == pw else None,
                             "ttest_p": round(pt, 4) if pt == pt else None,
                             "sig_wilcoxon": sig_marker(pw), "sig_ttest": sig_marker(pt)})
    pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "wilcoxon_stratified.csv"), index=False)

    # Table 5 -- retrieval metrics
    avail = [c for c in RETRIEVAL_COLS if c in df_main.columns]
    if avail:
        rows = []
        for mode in MODES_ORDERED:
            for qtype in [None] + TYPES_ORDERED:
                mask = df_main["mode"] == mode
                if qtype:
                    mask &= df_main["question_type"] == qtype
                sub = df_main[mask]
                if sub.empty:
                    continue
                row = {"mode": mode, "question_type": qtype or "ALL", "n": len(sub)}
                for c in avail:
                    row[c] = round(sub[c].mean(), 4)
                rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "retrieval.csv"), index=False)

    print(f"CSV tables written -> {tables_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse 2-mode evaluation results for Chapter 6")
    parser.add_argument("--results", nargs="+", default=DEFAULT_RESULTS,
                        help="One or more results CSVs to combine (default: 4 per-lane files)")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS,
                        help="Questions JSON for Precision@3 / Complete@3 enrichment")
    parser.add_argument("--output", default=None,
                        help="Write Markdown output to this file")
    parser.add_argument("--tables-dir", default=os.path.join("outputs", "tables"),
                        help="Directory to write per-table CSVs (default: outputs/tables/)")
    args = parser.parse_args()

    df = load_results(args.results, args.questions)

    df_main = df[df["mode"].isin(MODES_ORDERED)].copy()

    sections = [
        (
            f"# Experiment Analysis -- 2-Mode Evaluation\n\n"
            f"Sources: {args.results}\n"
            f"N success rows: {len(df)} total  |  {len(df_main)} in main comparison  |  "
            f"Modes: {sorted(df['mode'].unique())}  |  "
            f"Question types: {sorted(df['question_type'].unique())}\n"
        ),
        section_overall_means(df_main),
        section_per_type_means(df_main),
        section_wilcoxon_main(df_main),
        section_wilcoxon_stratified(df_main),
        section_retrieval(df_main),
        section_summary(df_main),
    ]

    out = "\n\n---\n\n".join(sections)
    print(out)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"\nAnalysis written -> {args.output}")

    export_tables_csv(df_main, args.tables_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
