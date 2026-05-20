from __future__ import annotations

import streamlit as st

from src.showcase.demo_runner import run_demo
from src.showcase.result_loader import load_gold_questions


st.set_page_config(
    page_title="AI Retrieval Inspector",
    page_icon="AI",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; }
    .small-muted { color: #64748b; font-size: 0.88rem; }
    .section-kicker {
        color: #38bdf8;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }
    .section-title {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 760;
        margin: 0.12rem 0 0.6rem 0;
    }
    .app-kicker {
        color: #38bdf8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .app-purpose {
        max-width: 920px;
        color: #94a3b8;
        font-size: 1rem;
        margin-top: -0.35rem;
        margin-bottom: 1.2rem;
    }
    .signal-card {
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.95rem 1rem;
        background: #0f172a;
        color: #e5e7eb;
        min-height: 96px;
    }
    .signal-label {
        color: #94a3b8;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .signal-value {
        color: #f8fafc;
        font-size: 1.45rem;
        line-height: 1.25;
        font-weight: 750;
        margin-top: 0.25rem;
    }
    .signal-note { color: #cbd5e1; font-size: 0.84rem; margin-top: 0.35rem; }
    .latency-badge {
        display: inline-block;
        border: 1px solid #334155;
        border-radius: 999px;
        background: #0f172a;
        color: #bae6fd;
        font-size: 0.78rem;
        padding: 0.15rem 0.55rem;
        margin-left: 0.35rem;
    }
    .confidence-bar {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        background: #1f2937;
        overflow: hidden;
        margin-top: 0.55rem;
    }
    .confidence-fill {
        height: 9px;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e, #38bdf8);
    }
    .inspector-panel {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        background: #111827;
        color: #e5e7eb;
        margin-bottom: 1rem;
    }
    .light-panel {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #ffffff;
        color: #0f172a !important;
    }
    .light-panel * { color: #0f172a !important; }
    .pipeline-step {
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        background: #111827;
        color: #e5e7eb;
        padding: 0.78rem 0.9rem;
        border-radius: 8px;
        min-height: 88px;
    }
    .pipeline-step strong { color: #f8fafc; }
    .pipeline-step .small-muted { color: #cbd5e1; }
    .flow-wrap {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.9rem;
        background: #0b1220;
        margin-top: 0.75rem;
    }
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.55rem;
        align-items: stretch;
    }
    .flow-node {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem;
        background: #111827;
        color: #e5e7eb;
        min-height: 76px;
        overflow-wrap: normal;
        word-break: normal;
    }
    .flow-node strong { color: #f8fafc; }
    .flow-arrow {
        color: #38bdf8;
        font-weight: 800;
        padding-left: 0.25rem;
    }
    .query-box {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: #0f172a;
        color: #e5e7eb;
        margin-bottom: 1rem;
    }
    .retriever-row {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        background: #ffffff;
        color: #0f172a !important;
        margin-bottom: 0.65rem;
    }
    .retriever-row * { color: #0f172a !important; }
    .chunk-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.7rem;
        background: #ffffff;
        color: #0f172a !important;
    }
    .chunk-card * { color: #0f172a !important; }
    .chunk-card strong { color: #0f172a !important; }
    .answer-panel {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        background: #0b1220;
        color: #e5e7eb;
    }
    .answer-panel code { color: #e5e7eb; }
    .risk-low { color: #047857; font-weight: 700; }
    .risk-moderate { color: #b45309; font-weight: 700; }
    .risk-elevated { color: #b91c1c; font-weight: 700; }
    .risk-panel {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
        color: #0f172a !important;
    }
    .risk-panel * { color: #0f172a !important; }
    .risk-panel .small-muted { color: #334155 !important; }
    .risk-panel .risk-low { color: #047857 !important; }
    .risk-panel .risk-moderate { color: #b45309 !important; }
    .risk-panel .risk-elevated { color: #b91c1c !important; }
    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        background: #ffffff;
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] * { color: #0f172a !important; }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_ms(value: float | int | str | None) -> str:
    try:
        return f"{float(value):,.2f} ms"
    except (TypeError, ValueError):
        return "not available"


def _risk_class(risk: str) -> str:
    return {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "Elevated": "risk-elevated",
    }.get(risk, "risk-moderate")


def _top_chunk_label(record: dict) -> str:
    contexts = record.get("retrieved_context") or []
    if not contexts:
        return "No evidence"
    top = contexts[0]
    return f"{top.get('source', 'unknown')} | chunk_id={top.get('chunk_id', 'unknown')}"


def render_section(number: int, title: str, detail: str | None = None) -> None:
    st.markdown(
        f"<div class='section-kicker'>Section {number}</div><div class='section-title'>{title}</div>",
        unsafe_allow_html=True,
    )
    if detail:
        st.caption(detail)


def render_signal_strip(run) -> None:
    latency = run.latency_ms or {}
    grounding = run.grounding or {}
    mode_label = "Saved benchmark run" if run.mode == "demo" else "Live inspection"
    risk = grounding.get("risk", "Moderate")
    citation_check = "grounded" if grounding.get("citations_grounded") else "not grounded"

    cols = st.columns(4)
    cards = [
        ("RUN MODE", mode_label, run.mode_detail),
        ("GROUNDING", f"{risk} risk", grounding.get("rationale", "")),
        ("CONFIDENCE", "not available", f"No calibrated confidence score exists in experiments/results. Citation check: {citation_check}."),
        ("TOTAL LATENCY", _format_ms(latency.get("total")), "Retrieval plus generation when available."),
    ]
    for col, (label, value, note) in zip(cols, cards):
        col.markdown(
            f"""
            <div class='signal-card'>
              <div class='signal-label'>{label}</div>
              <div class='signal-value'>{value}</div>
              <div class='signal-note'>{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_architecture_flow() -> None:
    labels = [
        ("Corpus", "Raw documents are chunked into source-aware passages."),
        ("Retriever layer", "Dense, BM25, hybrid, and dense-rerank expose competing evidence sets."),
        ("Answer layer", "The selected context is passed into the configured grounded-answer prompt."),
        ("Evaluation layer", "MRR, recall, citation grounding, unsupported risk, and latency are tracked."),
    ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, labels):
        col.markdown(
            f"<div class='pipeline-step'><strong>{title}</strong><br><span class='small-muted'>{body}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='flow-wrap'>
          <div class='flow-grid'>
            <div class='flow-node'><strong>Documents</strong><span class='flow-arrow'>-></span><br><span class='small-muted'>source chunks</span></div>
            <div class='flow-node'><strong>Retrievers</strong><span class='flow-arrow'>-></span><br><span class='small-muted'>dense / BM25 / hybrid / rerank</span></div>
            <div class='flow-node'><strong>Evidence</strong><span class='flow-arrow'>-></span><br><span class='small-muted'>ranked chunks and citations</span></div>
            <div class='flow-node'><strong>Answer</strong><span class='flow-arrow'>-></span><br><span class='small-muted'>grounded response artifact</span></div>
            <div class='flow-node'><strong>Risk</strong><br><span class='small-muted'>citation grounding and hallucination checks</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _retriever_overview_rows(run) -> list[dict[str, object]]:
    overview_rows = []
    for retriever_name in ["dense", "bm25", "hybrid", "dense_rerank"]:
        record = run.retrievers.get(retriever_name) or {}
        contexts = record.get("retrieved_context") or []
        latency = (record.get("latency_ms") or {}).get("retrieval")
        overview_rows.append(
            {
                "retriever": retriever_name,
                "evidence chunks": len(contexts),
                "top evidence": _top_chunk_label(record),
                "retrieval latency": _format_ms(latency),
                "run source": record.get("source", "saved experiment"),
            }
        )
    return overview_rows


def render_retriever_comparison(run) -> None:
    st.dataframe(_retriever_overview_rows(run), hide_index=True, use_container_width=True)


def render_evidence_explorer(run) -> None:
    tabs = st.tabs(["dense", "BM25", "hybrid", "dense_rerank"])
    for tab, retriever_name in zip(tabs, ["dense", "bm25", "hybrid", "dense_rerank"]):
        with tab:
            record = run.retrievers.get(retriever_name) or {}
            latency = (record.get("latency_ms") or {}).get("retrieval")
            st.markdown(
                f"**{retriever_name}** <span class='latency-badge'>{_format_ms(latency)}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"run source: {record.get('source', 'saved experiment')}")
            contexts = record.get("retrieved_context") or []
            if not contexts:
                st.info("No retrieved chunks available for this retriever.")
                continue
            for rank, chunk in enumerate(contexts[:5], start=1):
                source = chunk.get("source", "unknown")
                chunk_id = chunk.get("chunk_id", "unknown")
                text = " ".join(str(chunk.get("chunk_text", "")).split())
                with st.expander(f"Rank {rank} | {source} | chunk_id={chunk_id}", expanded=rank == 1):
                    st.markdown(text)


def render_answer(run) -> None:
    st.markdown("<div class='answer-panel'>", unsafe_allow_html=True)
    st.markdown(run.answer or "_No answer available._")
    st.markdown("</div>", unsafe_allow_html=True)

    if run.citations:
        st.markdown("**Citation trail**")
        for item in run.citations:
            st.write(f"- {item.get('source', 'unknown')} | chunk_id={item.get('chunk_id', 'unknown')}")


def render_latency_metrics(run) -> None:
    latency = run.latency_ms or {}
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval", _format_ms(latency.get("retrieval")))
    col2.metric("Generation", _format_ms(latency.get("generation")))
    col3.metric("Total", _format_ms(latency.get("total")))


def render_benchmark_insights(run) -> None:
    render_latency_metrics(run)
    summary_rows = run.metrics.get("retrieval", [])
    if summary_rows:
        st.markdown("**Retrieval quality summary**")
        st.dataframe(summary_rows, hide_index=True, use_container_width=True)

    scorecard = run.metrics.get("answer_scorecard", {}).get("experiments", [])
    if scorecard:
        rows = []
        for row in scorecard:
            name = "dense_rerank" if "dense_rerank" in row.get("input_path", "") else "dense"
            rows.append(
                {
                    "answer stack": name,
                    "correctness": row.get("avg_correctness_recall"),
                    "grounded citations": row.get("citations_grounded_rate"),
                    "unsupported risk": row.get("unsupported_risk_rate"),
                    "total avg ms": row.get("latency_ms", {}).get("total_avg"),
                }
            )
        st.markdown("**Answer quality and risk summary**")
        st.dataframe(rows, hide_index=True, use_container_width=True)


def render_grounding(run) -> None:
    grounding = run.grounding
    risk = grounding.get("risk", "Moderate")
    st.markdown(
        f"""
        <div class='risk-panel'>
          <div class='signal-label'>Hallucination-risk assessment</div>
          <div style='font-size:1.35rem; margin-top:0.25rem'><span class='{_risk_class(risk)}'>{risk} risk</span></div>
          <div style='color:#334155; font-size:0.9rem'>{grounding.get('rationale', '')}</div>
          <div style='color:#334155; font-size:0.9rem; margin-top:0.65rem'>Confidence score: not available in experiment outputs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    col1.metric("Citation count", grounding.get("citation_count", 0))
    col2.metric("Retrieved chunks checked", grounding.get("retrieved_chunk_count", 0))
    st.checkbox("Citations grounded in retrieved context", value=bool(grounding.get("citations_grounded")), disabled=True)


st.markdown("<div class='app-kicker'>Production RAG Observability</div>", unsafe_allow_html=True)
st.title("AI Retrieval Inspector")
st.markdown(
    "<div class='app-purpose'>Help developers understand why retrieval-augmented generation succeeds or fails by inspecting evidence, retriever behavior, grounding, latency, and hallucination risk.</div>",
    unsafe_allow_html=True,
)

gold_questions = load_gold_questions()
example_questions = [row["question"] for row in gold_questions[:12] if row.get("question")]

with st.sidebar:
    st.header("Inspection Controls")
    selected = st.selectbox("Example question", example_questions, index=0 if example_questions else None)
    use_selected = st.toggle("Use selected example", value=True)
    use_live_retrieval = st.toggle("Try live retrievers", value=False)
    use_live_generation = st.toggle("Try live LLM answer", value=False)
    top_k = st.slider("Top chunks", min_value=3, max_value=5, value=5)
    st.markdown(f"**Selected top chunks:** `{top_k}`")
    st.caption("Fallback mode uses saved benchmark artifacts as read-only evidence.")

render_section(1, "System Overview", "Current run health, architecture, grounding status, and latency at a glance.")

if "demo_run" not in st.session_state:
    initial_question = selected if use_selected and selected else "What does the Transformers library act as for modern ML models?"
    st.session_state["demo_run"] = run_demo(question=initial_question, use_live_retrieval=False, k=top_k)

run = st.session_state["demo_run"]

render_signal_strip(run)
render_architecture_flow()
st.divider()

render_section(2, "Query Input", "Select or enter the question whose retrieval behavior you want to inspect.")

question = st.text_area(
    "Inspection query",
    value=selected if use_selected and selected else "What does the Transformers library act as for modern ML models?",
    height=90,
)

if st.button("Run inspection", type="primary"):
    st.session_state["demo_run"] = run_demo(
        question=question,
        use_live_retrieval=use_live_retrieval,
        use_live_generation=use_live_generation,
        k=top_k,
    )

run = st.session_state["demo_run"]

if run.display_question != run.requested_question:
    st.caption(f"Showing saved demo question: {run.display_question}")

st.markdown(
    f"""
    <div class='query-box'>
      <div class='signal-label'>Current inspection target</div>
      <div style='font-size:1.05rem; margin-top:0.35rem'>{run.display_question}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

render_section(3, "Retriever Comparison", "Compare retrieval behavior across dense, BM25, hybrid, and dense-rerank.")
render_retriever_comparison(run)
st.divider()

render_section(4, "Retrieved Evidence Explorer", "Open ranked chunks to inspect the evidence each retriever surfaced.")
render_evidence_explorer(run)
st.divider()

left, right = st.columns([1.1, 1])
with left:
    render_section(5, "Answer + Citations", "View the final grounded answer artifact and its citation trail.")
    render_answer(run)
with right:
    render_section(6, "Grounding / Risk Analysis", "Check whether citations map back to retrieved chunks.")
    render_grounding(run)

st.divider()
render_section(7, "Benchmark Insights", "Use saved experiment metrics to reason about quality, latency, and production tradeoffs.")
render_benchmark_insights(run)
