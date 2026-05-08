"""
demo_app.py — Thesis Defense Demo: IU GraphRAG Academic Regulation Assistant

Three tabs:
  - Live Q&A         : interactive chat, compare-mode, sample question buttons
  - Results Dashboard: BERTScore/ROUGE-L/Recall@3 charts from experiment CSVs
  - Architecture     : pipeline overview, graph stats, key findings

Run:
    streamlit run demo_app.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Defense Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.finding-box {
    border-left: 4px solid #1976d2;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 10px 0;
    background: rgba(25, 118, 210, 0.1);
}
.result-row {
    padding: 8px 14px;
    border-radius: 4px;
    margin: 4px 0;
    font-family: monospace;
    font-size: 0.95rem;
}
.result-best  { background: rgba(40, 167, 69, 0.15);  border-left: 4px solid #28a745; }
.result-mid   { background: rgba(255, 193, 7, 0.15);  border-left: 4px solid #ffc107; }
.result-base  { background: rgba(108, 117, 125, 0.15); border-left: 4px solid #6c757d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODES = ["VECTOR_RETRIEVAL", "GRAPH_RETRIEVAL", "GRAPH_ITERATIVE"]
MODE_LABELS = {
    "VECTOR_RETRIEVAL": "VECTOR (baseline)",
    "GRAPH_RETRIEVAL":  "GRAPH (PPR)",
    "GRAPH_ITERATIVE":  "GRAPH+ITER (IRCoT)",
}
MODE_COLORS = {
    "VECTOR_RETRIEVAL": "#6c757d",
    "GRAPH_RETRIEVAL":  "#1976d2",
    "GRAPH_ITERATIVE":  "#28a745",
}
CONF_COLOR = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red", "NONE": "red"}
CITATION_RE = re.compile(r'\[(?:Article|Clause)\s*\d+[^\]]*\]', re.IGNORECASE)

RESULT_CSVS = [
    Path("outputs/eval_vec.csv"),
    Path("outputs/eval_graph.csv"),
    Path("outputs/eval_iter.csv"),
]

# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_data
def load_results() -> pd.DataFrame | None:
    frames = [pd.read_csv(p) for p in RESULT_CSVS if p.exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df[df["status"] == "success"].copy()


@st.cache_data
def load_questions() -> list:
    p = Path("data/test_set/questions_multihop_focused.json")
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading GraphRAG system (first run ~30 s)...")
def load_system():
    from src.core.single_groq_llm import SingleGroqLLM
    from src.engine.retriever import ThesisRetriever
    retriever = ThesisRetriever()
    llm = SingleGroqLLM(model="llama-3.1-8b-instant")
    retriever._reflect_llm = llm
    return retriever, llm

# ── Session state defaults ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("GraphRAG Demo")
    st.caption("IU Academic Regulation Q&A")
    st.divider()

    mode = st.selectbox(
        "Retrieval Mode",
        MODES,
        format_func=lambda m: MODE_LABELS[m],
        index=1,
        help=(
            "VECTOR: dense vector search only (baseline)\n"
            "GRAPH: PPR over L2+L3 graph + CrossEncoder filter\n"
            "GRAPH+ITER: GRAPH + one IRCoT reflection cycle"
        ),
    )

    compare_mode = st.toggle("Compare all 3 modes", value=False)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("LLM: llama-3.1-8b-instant (Groq)")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_qa, tab_results, tab_arch = st.tabs(["Live Q&A", "Results Dashboard", "Architecture"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE Q&A
# ══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.header("IU Academic Regulation Assistant")
    st.caption(
        "Ask questions about the undergraduate training regulation. "
        "The system retrieves relevant clauses from the knowledge graph and generates a cited answer."
    )

    questions = load_questions()
    if questions:
        st.markdown("**Quick questions from the evaluation set:**")
        btn_cols = st.columns(3)
        for i, q in enumerate(questions[:6]):
            label = q["query"][:58] + "..." if len(q["query"]) > 60 else q["query"]
            if btn_cols[i % 3].button(label, key=f"sq_{i}", use_container_width=True):
                st.session_state.prefill = q["query"]
                st.rerun()

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                c1, c2, c3 = st.columns([1, 1, 3])
                c1.caption(f"Mode: **{MODE_LABELS.get(meta.get('mode',''), meta.get('mode',''))}**")
                conf = meta.get("confidence", "?")
                c2.caption(f"Confidence: :{CONF_COLOR.get(conf, 'gray')}[{conf}]")
                if meta.get("citations"):
                    c3.caption("Citations: " + " · ".join(meta["citations"]))
                if meta.get("contexts"):
                    with st.expander("Retrieved context", expanded=False):
                        for idx, chunk in enumerate(meta["contexts"], 1):
                            st.markdown(f"**Chunk {idx}**")
                            st.code(chunk if isinstance(chunk, str) else str(chunk), language=None)

    GENERATION_PROMPT = (
        "You are an Academic Regulations Assistant. "
        "Answer the question based solely on the CONTEXT below.\n"
        "- Cite using [Article N] or [Clause N, Article M].\n"
        "- If not in context, say: 'I could not find this in the regulations.'\n"
        "- Be concise and factual.\n\n"
        "CONTEXT:\n{context}"
    )

    def _generate(context_str: str, query: str, llm) -> str:
        from llama_index.core.llms import ChatMessage
        msgs = [
            ChatMessage(role="system", content=GENERATION_PROMPT.format(context=context_str)),
            ChatMessage(role="user", content=query),
        ]
        try:
            return re.sub(r'\n{3,}', '\n\n', str(llm.chat(msgs))).strip()
        except Exception as exc:
            return f"Generation error: {exc}"

    def _run_single(query: str, sel_mode: str, retriever, llm) -> dict:
        try:
            context_str, confidence, chunks, _ = retriever.search(query, mode=sel_mode)
        except Exception as exc:
            return {"answer": f"Retrieval error: {exc}", "confidence": "NONE",
                    "citations": [], "contexts": [], "mode": sel_mode}
        if not context_str:
            return {"answer": "No relevant information found. (Check Neo4j is running and the clause_vector_index exists.)",
                    "confidence": "NONE", "citations": [], "contexts": [], "mode": sel_mode}
        answer = _generate(context_str, query, llm)
        citations = list(dict.fromkeys(CITATION_RE.findall(answer)))
        return {"answer": answer, "confidence": confidence,
                "citations": citations, "contexts": chunks, "mode": sel_mode}

    prompt = st.chat_input("Ask about the academic regulations...")
    if prompt is None and st.session_state.prefill:
        prompt = st.session_state.prefill
        st.session_state.prefill = None

    if prompt:
        retriever, llm = load_system()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if compare_mode:
            with st.chat_message("assistant"):
                st.markdown("**Comparing all 3 modes**")
                results = {}
                for m in MODES:
                    with st.spinner(f"Retrieving [{MODE_LABELS[m]}]..."):
                        results[m] = _run_single(prompt, m, retriever, llm)

                for m, res in results.items():
                    st.divider()
                    c_mode, c_conf = st.columns([3, 1])
                    c_mode.markdown(f"**{MODE_LABELS[m]}**")
                    conf = res["confidence"]
                    c_conf.caption(f":{CONF_COLOR.get(conf, 'gray')}[{conf}]")
                    st.markdown(res["answer"])
                    if res.get("citations"):
                        st.caption("Citations: " + " · ".join(res["citations"]))
                    if res.get("contexts"):
                        with st.expander(f"Retrieved context — {MODE_LABELS[m]}", expanded=False):
                            for j, chunk in enumerate(res["contexts"], 1):
                                st.markdown(f"**Chunk {j}**")
                                st.code(chunk if isinstance(chunk, str) else str(chunk), language=None)

                summary = "**Compare (3 modes):**\n\n" + "\n\n".join(
                    f"**{MODE_LABELS[m]}** [{r['confidence']}]: {r['answer'][:200]}..."
                    for m, r in results.items()
                )
                st.session_state.messages.append({
                    "role": "assistant", "content": summary,
                    "meta": {"mode": "COMPARE", "confidence": "—", "citations": [], "contexts": []},
                })
        else:
            with st.chat_message("assistant"):
                with st.spinner(f"Retrieving [{MODE_LABELS[mode]}]..."):
                    result = _run_single(prompt, mode, retriever, llm)
                st.markdown(result["answer"])
                c1, c2, c3 = st.columns([1, 1, 3])
                conf = result["confidence"]
                c1.caption(f"Mode: **{MODE_LABELS[mode]}**")
                c2.caption(f"Confidence: :{CONF_COLOR.get(conf, 'gray')}[{conf}]")
                if result.get("citations"):
                    c3.caption("Citations: " + " · ".join(result["citations"]))
                if result.get("contexts"):
                    with st.expander("Retrieved context", expanded=False):
                        for idx, chunk in enumerate(result["contexts"], 1):
                            st.markdown(f"**Chunk {idx}**")
                            st.code(chunk if isinstance(chunk, str) else str(chunk), language=None)
                st.session_state.messages.append({
                    "role": "assistant", "content": result["answer"], "meta": result,
                })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.header("Experiment Results Dashboard")
    st.caption(
        "3 modes x 28 questions (questions_multihop_focused.json). "
        "Metrics: BERTScore F1 + ROUGE-L (local, no API). One-tailed Wilcoxon signed-rank test."
    )

    df = load_results()
    if df is None:
        st.warning("No results CSVs found. Run the evaluation first.")
    else:
        st.markdown("""
        <div class="finding-box">
        <strong>Key findings:</strong>
        GRAPH_RETRIEVAL outperforms VECTOR_RETRIEVAL on BERTScore F1 (p=0.046*) and ROUGE-L (p=0.013*).
        GRAPH_ITERATIVE shows further ROUGE-L improvement (p=0.027* vs VECTOR).
        Recall@3 is lower for graph modes — PPR expands to adjacent unlabelled clauses (context quality paradox).
        </div>
        """, unsafe_allow_html=True)

        score_cols = [c for c in ["bertscore_f1", "rougeL", "recall@3"] if c in df.columns]
        summary = (
            df.groupby("mode")[score_cols].mean().round(3)
            .reindex([m for m in MODES if m in df["mode"].unique()])
        )

        metric_cols = st.columns(len(summary))
        for i, m in enumerate(summary.index):
            with metric_cols[i]:
                st.markdown(f"**{MODE_LABELS[m]}**")
                for col in score_cols:
                    st.markdown(f"`{col}`: **{summary.loc[m, col]:.3f}**")

        st.divider()

        try:
            import plotly.graph_objects as go

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("BERTScore F1 by Mode")
                fig = go.Figure(go.Bar(
                    x=[MODE_LABELS[m] for m in summary.index],
                    y=summary["bertscore_f1"].tolist() if "bertscore_f1" in summary.columns else [],
                    marker_color=[MODE_COLORS.get(m, "#888") for m in summary.index],
                    text=[f"{v:.3f}" for v in summary["bertscore_f1"]] if "bertscore_f1" in summary.columns else [],
                    textposition="outside",
                ))
                fig.update_layout(
                    yaxis_range=[0.55, 0.75], height=360, margin=dict(t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("ROUGE-L by Mode")
                fig2 = go.Figure(go.Bar(
                    x=[MODE_LABELS[m] for m in summary.index],
                    y=summary["rougeL"].tolist() if "rougeL" in summary.columns else [],
                    marker_color=[MODE_COLORS.get(m, "#888") for m in summary.index],
                    text=[f"{v:.3f}" for v in summary["rougeL"]] if "rougeL" in summary.columns else [],
                    textposition="outside",
                ))
                fig2.update_layout(
                    yaxis_range=[0.2, 0.45], height=360, margin=dict(t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2, use_container_width=True)

            if "bertscore_f1" in df.columns and "rougeL" in df.columns:
                st.subheader("Per-question BERTScore F1 vs ROUGE-L")
                fig_sc = go.Figure()
                for m in MODES:
                    sub = df[df["mode"] == m]
                    if sub.empty:
                        continue
                    fig_sc.add_trace(go.Scatter(
                        x=sub["rougeL"], y=sub["bertscore_f1"],
                        mode="markers", name=MODE_LABELS[m],
                        marker=dict(color=MODE_COLORS.get(m, "#888"), size=9),
                        text=sub["question_id"],
                        hovertemplate="<b>%{text}</b><br>ROUGE-L=%{x:.3f}  BERTScore=%{y:.3f}<extra></extra>",
                    ))
                fig_sc.update_layout(
                    xaxis_title="ROUGE-L", yaxis_title="BERTScore F1", height=420,
                    margin=dict(t=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sc, use_container_width=True)

        except ModuleNotFoundError:
            st.subheader("BERTScore F1 by Mode")
            st.bar_chart(summary["bertscore_f1"] if "bertscore_f1" in summary.columns else summary)
            st.subheader("ROUGE-L by Mode")
            st.bar_chart(summary["rougeL"] if "rougeL" in summary.columns else summary)

        if "question_type" in df.columns:
            st.subheader("BERTScore F1 by Question Type x Mode")
            pivot = (
                df.groupby(["question_type", "mode"])["bertscore_f1"]
                .mean().unstack().round(3)
                .reindex(columns=[m for m in MODES if m in df["mode"].unique()])
            )
            pivot.columns = [MODE_LABELS.get(c, c) for c in pivot.columns]
            st.dataframe(pivot, use_container_width=True)

        with st.expander("Full results table", expanded=False):
            show_cols = [c for c in
                ["question_id", "question_type", "mode", "bertscore_f1", "rougeL",
                 "recall@3", "hit@3", "confidence", "latency_ms", "reflection_gap"]
                if c in df.columns]
            st.dataframe(
                df[show_cols].sort_values(["question_id", "mode"]),
                use_container_width=True, height=400,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.header("System Architecture")

    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("4-Layer Knowledge Graph Pipeline")
        st.markdown("""
| Layer | Name | Content |
|---|---|---|
| **L1** | Structural | Document → Chapter → Article → Clause hierarchy |
| **L2** | Cross-references | `REFERS_TO` edges between clauses (regex-extracted) |
| **L3** | Semantic entities | `MENTIONS`, `REQUIRES`, `LEADS_TO`, `AFFECTS`, `HAS_CONDITION`, `EXEMPTS`, `PERFORMS` |
| **L4** | Vector index | `all-MiniLM-L6-v2` embeddings on all Clause nodes (384-dim, cosine) |
        """)

        st.subheader("3 Retrieval Modes (Thesis Evaluation Design)")
        st.markdown("""
| Mode | Strategy | Role |
|---|---|---|
| **VECTOR_RETRIEVAL** | Dense ANN (score > 0.5) + top-3 | Baseline |
| **GRAPH_RETRIEVAL** | PPR (alpha=0.85) over L2+L3, 70/30 anchor/entity seeding, CrossEncoder filter | Proposed method |
| **GRAPH_ITERATIVE** | GRAPH_RETRIEVAL + 1-cycle IRCoT reflection (gap fill) | Iterative extension |
        """)

        st.subheader("Evaluation Protocol")
        st.markdown("""
- **Test set:** 28 questions — 14 CROSSREF + 14 MULTIHOP
- **Metrics:** BERTScore F1 (`bert-base-uncased`, local) · ROUGE-L · Recall@3 · Hit@3
- **Statistical test:** Paired Wilcoxon signed-rank (one-tailed, alpha=0.05)
- **LLM:** `llama-3.1-8b-instant` via Groq (pinned — no substitution)
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (CPU)
- **Total cells:** 3 modes × 28 questions = 84
        """)

    with c2:
        st.subheader("Graph Statistics")
        st.markdown("""
| Item | Count |
|---|---|
| Clause nodes | 117 |
| Entity nodes | 474 |
| MENTIONS edges | 834 |
| REFERS_TO edges | 28 |
| Typed L3 relation types | 6 |
| CO_OCCURS edges (excluded from PPR) | 2308 |
        """)

        st.subheader("Quantitative Results (N=28)")
        st.markdown("""
<div class="result-row result-mid">VECTOR &nbsp;&nbsp;&nbsp;— BERTScore: 0.623 · ROUGE-L: 0.278 · Recall@3: 0.619 (baseline)</div>
<div class="result-row result-best">GRAPH &nbsp;&nbsp;&nbsp;&nbsp;— BERTScore: 0.650 · ROUGE-L: 0.324 · Recall@3: 0.577 (p=0.046* / p=0.013*)</div>
<div class="result-row result-best">ITER &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— BERTScore: 0.660 · ROUGE-L: 0.329 · Recall@3: 0.577 (ROUGE-L p=0.027*)</div>
""", unsafe_allow_html=True)

        st.subheader("Why does Recall@3 decrease for graph modes?")
        st.markdown("""
PPR expands to adjacent regulatory clauses not listed in the gold-standard
`source_clause_id` annotations. These clauses are semantically relevant
(e.g. grading tables, definitions referenced by the target clause) but
are not annotated as sources, so they count as misses. Generation quality
(BERTScore / ROUGE-L) improves precisely because this expanded context
contains the information needed to answer correctly.
        """)

        st.subheader("Tech Stack")
        st.markdown("""
- **Graph DB:** Neo4j 5 (local)
- **Framework:** LlamaIndex (llama-index-core 0.14)
- **Graph construction LLM:** FallbackGroqLLM (llama-3.3-70b → qwen3-32b → llama-3.1-8b)
- **Generation / reflection LLM:** SingleGroqLLM (llama-3.1-8b-instant, fixed)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **UI:** Streamlit
        """)
