import argparse
import json
from pathlib import Path

from src.common.config import load_config


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    """Save a list of dictionaries to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: dict, path: str | Path) -> None:
    """Save a dictionary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recall_at_k(expected_chunk_ids: list[int], retrieved_chunk_ids: list[int], k: int) -> float:
    """Calculate Recall@K for a single question."""
    expected = set(expected_chunk_ids)
    retrieved = set(retrieved_chunk_ids[:k])

    if not expected:
        return 0.0

    hits = len(expected.intersection(retrieved))
    return hits / len(expected)


def reciprocal_rank(expected_chunk_ids: list[int], retrieved_chunk_ids: list[int]) -> float:
    """Calculate reciprocal rank for a single question."""
    expected = set(expected_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank

    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval outputs.")
    parser.add_argument(
        "--config",
        default="configs/dense.yaml",
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate retrieval performance using config-driven inputs and outputs."""
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)

    gold_dataset_path = config["paths"]["gold_dataset_path"]
    predictions_path = config["paths"]["predictions_output"]
    retrieval_eval_output = config["paths"]["retrieval_eval_output"]
    retrieval_summary_output = config["paths"]["retrieval_summary_output"]

    experiment_name = config["experiment"]["name"]
    retriever_type = config["retrieval"]["type"]
    k_values = config["evaluation"]["retrieval_k_values"]

    gold_rows = load_jsonl(gold_dataset_path)
    pred_rows = load_jsonl(predictions_path)

    pred_by_question = {row["question"]: row for row in pred_rows}

    per_question_results = []
    recall_totals = {f"recall_at_{k}": 0.0 for k in k_values}
    total_mrr = 0.0
    count = 0

    for gold in gold_rows:
        question = gold["question"]
        expected_chunk_ids = gold["expected_chunk_ids"]

        pred = pred_by_question.get(question)
        if pred is None:
            print(f"Missing prediction for question: {question}")
            continue

        retrieved_chunk_ids = [
            item["chunk_id"]
            for item in pred["retrieved_context"]
            if item.get("chunk_id") != "unknown"
        ]

        recall_scores = {
            f"recall_at_{k}": round(recall_at_k(expected_chunk_ids, retrieved_chunk_ids, k=k), 3)
            for k in k_values
        }
        rr = round(reciprocal_rank(expected_chunk_ids, retrieved_chunk_ids), 3)

        result_row = {
            "question": question,
            "expected_chunk_ids": expected_chunk_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            **recall_scores,
            "reciprocal_rank": rr,
        }
        per_question_results.append(result_row)

        for k in k_values:
            recall_totals[f"recall_at_{k}"] += recall_scores[f"recall_at_{k}"]
        total_mrr += rr
        count += 1

    if count == 0:
        print("No questions were evaluated.")
        return

    summary = {
        "experiment_name": experiment_name,
        "retriever_type": retriever_type,
        "gold_dataset_path": gold_dataset_path,
        "predictions_path": predictions_path,
        "num_questions": count,
        "k_values": k_values,
        "mrr": round(total_mrr / count, 3),
    }
    for metric_name, total in recall_totals.items():
        summary[f"avg_{metric_name}"] = round(total / count, 3)

    save_jsonl(per_question_results, retrieval_eval_output)
    save_json(summary, retrieval_summary_output)

    print(f"Experiment: {experiment_name}")
    print(f"Retrieval Type: {retriever_type}")
    print(f"Gold Dataset: {gold_dataset_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Saved retrieval evaluation to {retrieval_eval_output}")
    print(f"Saved summary to {retrieval_summary_output}")


if __name__ == "__main__":
    main()
