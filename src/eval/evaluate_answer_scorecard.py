from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple answer quality and latency scorecard from prediction JSONL files."
    )
    parser.add_argument(
        "--gold",
        default="data/eval/gold_qa.jsonl",
        help="Path to the gold dataset JSONL.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more answer-generation JSONL files to score.",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/answer_scorecard_summary.json",
        help="Path to save the scorecard JSON.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(data: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def normalize_key(source: str, chunk_id: int) -> str:
    return f"{str(source).replace('\\', '/')}::{int(chunk_id)}"


def avg(values: list[float]) -> float:
    return round(mean(values), 3) if values else 0.0


def med(values: list[float]) -> float:
    return round(median(values), 2) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round(0.95 * len(sorted_values)) - 1))
    return round(sorted_values[index], 2)


def build_gold_index(gold_rows: list[dict]) -> dict[str, dict]:
    return {row["question"]: row for row in gold_rows}


def score_prediction(prediction: dict, gold: dict) -> dict:
    answer = prediction.get("answer", "")
    reference_answer = gold.get("reference_answer", "")

    answer_tokens = tokenize(answer)
    reference_tokens = tokenize(reference_answer)
    if reference_tokens:
        correctness_recall = len(answer_tokens & reference_tokens) / len(reference_tokens)
    else:
        correctness_recall = 0.0

    retrieved_context = prediction.get("retrieved_context", [])
    citations = prediction.get("citations", [])

    retrieved_keys = {
        normalize_key(item.get("source", "unknown"), item.get("chunk_id", -1))
        for item in retrieved_context
        if item.get("chunk_id") != "unknown"
    }
    citation_keys = {
        normalize_key(item.get("source", "unknown"), item.get("chunk_id", -1))
        for item in citations
        if item.get("chunk_id") != "unknown"
    }

    expected_sources = gold.get("expected_sources", [])
    expected_chunk_ids = gold.get("expected_chunk_ids", [])
    if len(expected_sources) == len(expected_chunk_ids):
        expected_keys = {
            normalize_key(source, chunk_id)
            for source, chunk_id in zip(expected_sources, expected_chunk_ids)
        }
    else:
        expected_keys = {
            normalize_key(source, chunk_id)
            for source in expected_sources
            for chunk_id in expected_chunk_ids
        }

    expected_chunk_retrieved = bool(expected_keys & retrieved_keys)
    citations_grounded = bool(citation_keys) and citation_keys.issubset(retrieved_keys)

    answer_lower = answer.lower()
    abstained = (
        "not enough information" in answer_lower
        or "i don't have enough information" in answer_lower
        or "cannot answer" in answer_lower
    )
    unsupported_risk = bool(answer.strip()) and not expected_chunk_retrieved and not abstained

    latency = prediction.get("latency_ms", {})

    return {
        "question_id": gold.get("question_id"),
        "question": gold["question"],
        "correctness_recall": round(correctness_recall, 3),
        "expected_chunk_retrieved": expected_chunk_retrieved,
        "citations_grounded": citations_grounded,
        "abstained": abstained,
        "unsupported_risk": unsupported_risk,
        "latency_ms": {
            "retrieval": float(latency.get("retrieval", 0.0)),
            "generation": float(latency.get("generation", 0.0)),
            "total": float(latency.get("total", 0.0)),
        },
    }


def summarize_run(path: str | Path, gold_index: dict[str, dict]) -> dict:
    predictions = load_jsonl(path)
    scored_rows = [
        score_prediction(prediction, gold_index[prediction["question"]])
        for prediction in predictions
        if prediction["question"] in gold_index
    ]

    retrieval_latencies = [row["latency_ms"]["retrieval"] for row in scored_rows]
    generation_latencies = [row["latency_ms"]["generation"] for row in scored_rows]
    total_latencies = [row["latency_ms"]["total"] for row in scored_rows]

    return {
        "input_path": str(path),
        "num_records": len(scored_rows),
        "avg_correctness_recall": avg([row["correctness_recall"] for row in scored_rows]),
        "expected_chunk_retrieved_rate": avg(
            [1.0 if row["expected_chunk_retrieved"] else 0.0 for row in scored_rows]
        ),
        "citations_grounded_rate": avg(
            [1.0 if row["citations_grounded"] else 0.0 for row in scored_rows]
        ),
        "abstention_rate": avg([1.0 if row["abstained"] else 0.0 for row in scored_rows]),
        "unsupported_risk_rate": avg(
            [1.0 if row["unsupported_risk"] else 0.0 for row in scored_rows]
        ),
        "latency_ms": {
            "retrieval_avg": round(mean(retrieval_latencies), 2) if retrieval_latencies else 0.0,
            "retrieval_median": med(retrieval_latencies),
            "retrieval_p95": p95(retrieval_latencies),
            "generation_avg": round(mean(generation_latencies), 2) if generation_latencies else 0.0,
            "generation_median": med(generation_latencies),
            "generation_p95": p95(generation_latencies),
            "total_avg": round(mean(total_latencies), 2) if total_latencies else 0.0,
            "total_median": med(total_latencies),
            "total_p95": p95(total_latencies),
        },
    }


def main() -> None:
    args = parse_args()
    gold_index = build_gold_index(load_jsonl(args.gold))
    summaries = [summarize_run(path, gold_index) for path in args.inputs]
    payload = {"experiments": summaries}

    save_json(payload, args.output)
    print(f"Saved answer scorecard to {args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
