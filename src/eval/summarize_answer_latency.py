from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize latency metrics from answer-generation JSONL files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more answer-generation JSONL files to summarize.",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/answer_latency_summary.json",
        help="Path to save the latency summary JSON.",
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


def mean_or_zero(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    return round(median(values), 2) if values else 0.0


def p95_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, int(0.95 * len(sorted_values)) - 1)
    return round(sorted_values[index], 2)


def summarize_file(path: str | Path) -> dict:
    rows = load_jsonl(path)

    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    populated_answers = 0
    populated_citations = 0

    for row in rows:
        latency = row.get("latency_ms", {})
        retrieval_latencies.append(float(latency.get("retrieval", 0.0)))
        generation_latencies.append(float(latency.get("generation", 0.0)))
        total_latencies.append(float(latency.get("total", 0.0)))

        if row.get("answer"):
            populated_answers += 1
        if row.get("citations"):
            populated_citations += 1

    return {
        "input_path": str(path),
        "num_records": len(rows),
        "answer_populated_count": populated_answers,
        "citations_populated_count": populated_citations,
        "latency_ms": {
            "retrieval_avg": mean_or_zero(retrieval_latencies),
            "retrieval_median": median_or_zero(retrieval_latencies),
            "retrieval_p95": p95_or_zero(retrieval_latencies),
            "generation_avg": mean_or_zero(generation_latencies),
            "generation_median": median_or_zero(generation_latencies),
            "generation_p95": p95_or_zero(generation_latencies),
            "total_avg": mean_or_zero(total_latencies),
            "total_median": median_or_zero(total_latencies),
            "total_p95": p95_or_zero(total_latencies),
        },
    }


def save_json(data: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    summaries = [summarize_file(path) for path in args.inputs]
    payload = {"experiments": summaries}

    save_json(payload, args.output)
    print(f"Saved latency summary to {args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
