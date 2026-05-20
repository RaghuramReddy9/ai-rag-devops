from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
GOLD_PATH = PROJECT_ROOT / "data" / "eval" / "gold_qa.jsonl"

RETRIEVER_RESULT_FILES = {
    "dense": RESULTS_DIR / "dense_predictions.jsonl",
    "bm25": RESULTS_DIR / "bm25_predictions.jsonl",
    "hybrid": RESULTS_DIR / "hybrid_predictions.jsonl",
    "dense_rerank": RESULTS_DIR / "dense_rerank_predictions.jsonl",
}

ANSWER_RESULT_FILES = {
    "dense": RESULTS_DIR / "dense_answers.jsonl",
    "dense_rerank": RESULTS_DIR / "dense_rerank_answers.jsonl",
}

SUMMARY_FILES = {
    "dense": RESULTS_DIR / "dense_retrieval_summary.json",
    "bm25": RESULTS_DIR / "bm25_retrieval_summary.json",
    "hybrid": RESULTS_DIR / "hybrid_retrieval_summary.json",
    "dense_rerank": RESULTS_DIR / "dense_rerank_retrieval_summary.json",
}


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []

    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold_questions() -> list[dict[str, Any]]:
    return load_jsonl(GOLD_PATH)


def load_retrieval_summaries() -> dict[str, dict[str, Any]]:
    return {name: load_json(path) for name, path in SUMMARY_FILES.items()}


def load_answer_scorecard() -> dict[str, Any]:
    return load_json(RESULTS_DIR / "final_answer_scorecard_summary.json")


def load_answer_latency() -> dict[str, Any]:
    return load_json(RESULTS_DIR / "final_answer_latency_summary.json")


def load_prediction_records() -> dict[str, list[dict[str, Any]]]:
    return {name: load_jsonl(path) for name, path in RETRIEVER_RESULT_FILES.items()}


def load_answer_records() -> dict[str, list[dict[str, Any]]]:
    return {name: load_jsonl(path) for name, path in ANSWER_RESULT_FILES.items()}


def normalize_source(source: Any) -> str:
    return str(source or "unknown").replace("\\", "/")


def citation_key(item: dict[str, Any]) -> str:
    return f"{normalize_source(item.get('source'))}::{item.get('chunk_id', 'unknown')}"


def find_exact_question(records: list[dict[str, Any]], question: str) -> dict[str, Any] | None:
    normalized = question.strip().casefold()
    for row in records:
        if str(row.get("question", "")).strip().casefold() == normalized:
            return row
    return None


def choose_demo_question(question: str, gold_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not gold_rows:
        return None

    exact = find_exact_question(gold_rows, question)
    if exact:
        return exact

    query_tokens = set(re.findall(r"\b\w+\b", question.casefold()))
    if not query_tokens:
        return gold_rows[0]

    def overlap(row: dict[str, Any]) -> int:
        row_tokens = set(re.findall(r"\b\w+\b", str(row.get("question", "")).casefold()))
        return len(query_tokens & row_tokens)

    return max(gold_rows, key=overlap)


def get_summary_table() -> list[dict[str, Any]]:
    summaries = load_retrieval_summaries()
    rows: list[dict[str, Any]] = []
    for name, summary in summaries.items():
        if not summary:
            continue
        rows.append(
            {
                "retriever": name,
                "MRR": summary.get("mrr", "not available"),
                "Recall@1": summary.get("avg_recall_at_1", "not available"),
                "Recall@3": summary.get("avg_recall_at_3", "not available"),
                "Recall@5": summary.get("avg_recall_at_5", "not available"),
            }
        )
    return rows
