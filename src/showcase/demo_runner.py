from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.common.config import load_config
from src.showcase.result_loader import (
    citation_key,
    choose_demo_question,
    find_exact_question,
    get_summary_table,
    load_answer_latency,
    load_answer_records,
    load_answer_scorecard,
    load_gold_questions,
    load_prediction_records,
)


RETRIEVER_CONFIGS = {
    "dense": "configs/dense.yaml",
    "bm25": "configs/bm25.yaml",
    "hybrid": "configs/hybrid.yaml",
    "dense_rerank": "configs/dense_rerank.yaml",
}


@dataclass
class DemoRun:
    question: str
    display_question: str
    requested_question: str
    mode: str
    mode_detail: str
    retrievers: dict[str, dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    latency_ms: dict[str, float]
    grounding: dict[str, Any]
    metrics: dict[str, Any]


def _doc_to_context(doc: Any) -> dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "source": metadata.get("source", "unknown"),
        "chunk_id": metadata.get("chunk_id", "unknown"),
        "chunk_text": doc.page_content,
    }


def _record_for_question(records: list[dict[str, Any]], question: str) -> dict[str, Any]:
    return find_exact_question(records, question) or {}


def _run_live_retrieval(question: str, k: int) -> tuple[dict[str, dict[str, Any]], str | None]:
    from src.retrieval.factory import build_retriever

    results: dict[str, dict[str, Any]] = {}
    for retriever_name, config_path in RETRIEVER_CONFIGS.items():
        config = load_config(config_path)
        top_k = int(config.get("retrieval", {}).get("top_k", k))
        if retriever_name != "dense_rerank":
            top_k = k

        start = perf_counter()
        retriever = build_retriever(config)
        docs = retriever.retrieve(question, k=top_k)
        elapsed = round((perf_counter() - start) * 1000, 2)
        context = [_doc_to_context(doc) for doc in docs]
        results[retriever_name] = {
            "retriever": retriever_name,
            "question": question,
            "retrieved_context": context,
            "citations": [
                {"source": item["source"], "chunk_id": item["chunk_id"]}
                for item in context
            ],
            "latency_ms": {"retrieval": elapsed, "generation": 0.0, "total": elapsed},
            "source": "live",
        }
    return results, None


def _fallback_retrieval(question: str) -> tuple[str, dict[str, dict[str, Any]], str]:
    gold_rows = load_gold_questions()
    demo_question = choose_demo_question(question, gold_rows)
    display_question = str((demo_question or {}).get("question") or question)
    prediction_records = load_prediction_records()

    results: dict[str, dict[str, Any]] = {}
    for retriever_name, records in prediction_records.items():
        record = _record_for_question(records, display_question)
        if record:
            record = dict(record)
            record["source"] = "saved experiment"
        results[retriever_name] = record

    detail = "Using saved benchmark artifacts because live retrieval is unavailable."
    if display_question != question:
        detail += f" Closest demo question: {display_question}"
    return display_question, results, detail


def _answer_from_saved(display_question: str) -> dict[str, Any]:
    answer_records = load_answer_records()
    for retriever_name in ("dense_rerank", "dense"):
        record = _record_for_question(answer_records.get(retriever_name, []), display_question)
        if record:
            return dict(record)
    return {}


def _generate_live_answer(question: str, context: list[dict[str, Any]]) -> tuple[str, float, str | None]:
    if not os.getenv("OPENROUTER_API_KEY"):
        return "", 0.0, "Missing OPENROUTER_API_KEY; using saved answer artifact when available."

    from langchain_core.documents import Document

    from src.generation.generator import generate_answer, get_llm, load_prompt

    docs = [
        Document(
            page_content=str(item.get("chunk_text", "")),
            metadata={"source": item.get("source"), "chunk_id": item.get("chunk_id")},
        )
        for item in context
    ]
    start = perf_counter()
    llm = get_llm(model_name=load_config()["models"]["llm_model"])
    prompt = load_prompt(config_path=load_config()["paths"]["prompt_config"])
    answer = generate_answer(question, docs, llm=llm, prompt=prompt)
    return answer, round((perf_counter() - start) * 1000, 2), None


def _grounding_panel(answer_record: dict[str, Any], preferred_context: list[dict[str, Any]]) -> dict[str, Any]:
    citations = answer_record.get("citations", [])
    retrieved_keys = {citation_key(item) for item in preferred_context}
    citation_keys = {citation_key(item) for item in citations}
    answer = str(answer_record.get("answer", ""))
    grounded = bool(citation_keys) and citation_keys.issubset(retrieved_keys)
    abstained = "not enough information" in answer.casefold() or "cannot answer" in answer.casefold()

    if grounded and not abstained:
        risk = "Low"
        rationale = "Citations are present and map to retrieved chunks."
    elif abstained:
        risk = "Moderate"
        rationale = "The answer abstains or hedges, which reduces hallucination risk but may be incomplete."
    else:
        risk = "Elevated"
        rationale = "Citations are missing or do not fully map to retrieved chunks."

    return {
        "risk": risk,
        "citations_grounded": grounded,
        "citation_count": len(citation_keys),
        "retrieved_chunk_count": len(retrieved_keys),
        "rationale": rationale,
    }


def _metrics() -> dict[str, Any]:
    return {
        "retrieval": get_summary_table(),
        "answer_scorecard": load_answer_scorecard(),
        "answer_latency": load_answer_latency(),
    }


def run_demo(question: str, use_live_retrieval: bool = True, use_live_generation: bool = False, k: int = 5) -> DemoRun:
    requested_question = question.strip()
    if not requested_question:
        requested_question = "What does the Transformers library act as for modern ML models?"

    display_question = requested_question
    mode = "live"
    mode_detail = "Live retriever call succeeded."

    if use_live_retrieval:
        try:
            retrievers, _ = _run_live_retrieval(requested_question, k=k)
        except Exception as exc:
            display_question, retrievers, detail = _fallback_retrieval(requested_question)
            mode = "demo"
            mode_detail = f"{detail} ({type(exc).__name__}: {exc})"
    else:
        display_question, retrievers, mode_detail = _fallback_retrieval(requested_question)
        mode = "demo"

    preferred = retrievers.get("dense_rerank") or retrievers.get("dense") or {}
    preferred_context = preferred.get("retrieved_context", [])
    answer_record: dict[str, Any] = {}

    generation_latency = 0.0
    if mode == "live" and use_live_generation:
        try:
            answer, generation_latency, generation_error = _generate_live_answer(
                requested_question,
                preferred_context,
            )
            if answer:
                answer_record = {
                    "answer": answer,
                    "citations": preferred.get("citations", []),
                    "retrieved_context": preferred_context,
                }
            elif generation_error:
                mode_detail += f" {generation_error}"
        except Exception as exc:
            mode_detail += f" Live generation failed; using grounded fallback answer. ({type(exc).__name__}: {exc})"

    if not answer_record:
        answer_record = _answer_from_saved(display_question)

    if not answer_record:
        answer_record = {
            "answer": "not available",
            "citations": [],
            "retrieved_context": preferred_context,
        }
        mode_detail += " No saved answer-generation artifact was available for this question."

    retrieval_latency = float(preferred.get("latency_ms", {}).get("retrieval", 0.0) or 0.0)
    latency = dict(answer_record.get("latency_ms") or {})
    if not latency:
        latency = {"retrieval": retrieval_latency} if retrieval_latency else {}

    return DemoRun(
        question=display_question,
        display_question=display_question,
        requested_question=requested_question,
        mode=mode,
        mode_detail=mode_detail,
        retrievers=retrievers,
        answer=str(answer_record.get("answer", "")),
        citations=answer_record.get("citations", []),
        latency_ms=latency,
        grounding=_grounding_panel(answer_record, preferred_context),
        metrics=_metrics(),
    )
