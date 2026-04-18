import argparse
import json
from pathlib import Path
from statistics import mean

from src.common.config import load_config
from src.eval.prediction_runner import load_gold_questions
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import DenseRetriever


QUESTION_CATEGORY_MAP = {
    "gold_001": "direct",
    "gold_002": "direct",
    "gold_003": "direct",
    "gold_004": "direct",
    "gold_005": "direct",
    "gold_006": "direct",
    "gold_007": "direct",
    "gold_008": "direct",
    "gold_009": "direct",
    "gold_010": "direct",
    "gold_011": "keyword_heavy",
    "gold_012": "keyword_heavy",
    "gold_013": "keyword_heavy",
    "gold_014": "keyword_heavy",
    "gold_015": "keyword_heavy",
    "gold_016": "keyword_heavy",
    "gold_017": "keyword_heavy",
    "gold_018": "keyword_heavy",
    "gold_019": "keyword_heavy",
    "gold_020": "keyword_heavy",
    "gold_021": "paraphrased_semantic",
    "gold_022": "paraphrased_semantic",
    "gold_023": "paraphrased_semantic",
    "gold_024": "paraphrased_semantic",
    "gold_025": "paraphrased_semantic",
    "gold_026": "paraphrased_semantic",
    "gold_027": "paraphrased_semantic",
    "gold_028": "paraphrased_semantic",
    "gold_029": "paraphrased_semantic",
    "gold_030": "paraphrased_semantic",
    "gold_031": "paraphrased_semantic",
    "gold_032": "paraphrased_semantic",
    "gold_033": "multi_evidence",
    "gold_034": "multi_evidence",
    "gold_035": "multi_evidence",
    "gold_036": "multi_evidence",
    "gold_037": "multi_evidence",
    "gold_038": "multi_evidence",
    "gold_039": "multi_evidence",
    "gold_040": "multi_evidence",
    "gold_041": "distractor",
    "gold_042": "distractor",
    "gold_043": "distractor",
    "gold_044": "distractor",
    "gold_045": "distractor",
    "gold_046": "distractor",
    "gold_047": "distractor",
    "gold_048": "distractor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze overlap between dense and BM25 retrieval results."
    )
    parser.add_argument(
        "--dense-config",
        default="configs/dense.yaml",
        help="Path to the dense experiment config file.",
    )
    parser.add_argument(
        "--bm25-config",
        default="configs/bm25.yaml",
        help="Path to the BM25 experiment config file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k depth to compare for dense and BM25 retrieval.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="experiments/results/retrieval_overlap_analysis.jsonl",
        help="Path to save per-query overlap analysis.",
    )
    parser.add_argument(
        "--summary-output",
        default="experiments/results/retrieval_overlap_summary.json",
        help="Path to save aggregate overlap metrics.",
    )
    return parser.parse_args()


def make_key(source: str, chunk_id: int) -> str:
    normalized_source = str(source).replace("\\", "/")
    return f"{normalized_source}::{int(chunk_id)}"


def document_to_key(document) -> str:
    metadata = document.metadata or {}
    return make_key(metadata["source"], metadata["chunk_id"])


def save_jsonl(rows: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def pct_true(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return round(100.0 * sum(flags) / len(flags), 1)


def build_category_summary(rows: list[dict]) -> dict:
    categories = sorted({row["question_category"] for row in rows})
    summary = {}

    for category in categories:
        category_rows = [row for row in rows if row["question_category"] == category]
        summary[category] = {
            "num_questions": len(category_rows),
            "avg_overlap_at_k": round(
                mean(row["intersection_size"] for row in category_rows), 3
            )
            if category_rows
            else 0.0,
            "pct_queries_where_bm25_adds_new_relevant_chunk": pct_true(
                [row["bm25_adds_new_relevant_chunk"] for row in category_rows]
            ),
            "pct_queries_where_bm25_is_useless": pct_true(
                [row["bm25_is_useless"] for row in category_rows]
            ),
            "bm25_helpful_question_ids": [
                row["question_id"]
                for row in category_rows
                if row["bm25_adds_new_relevant_chunk"]
            ],
        }

    return summary


def main() -> None:
    args = parse_args()

    dense_config = load_config(args.dense_config)
    bm25_config = load_config(args.bm25_config)
    gold_rows = load_gold_questions(dense_config["paths"]["gold_dataset_path"])

    dense_retriever = DenseRetriever(persist_dir=dense_config["paths"]["persist_dir"])
    bm25_retriever = BM25Retriever(chunks_path=bm25_config["paths"]["chunks_output"])

    per_query_rows = []
    overlap_sizes = []
    bm25_adds_new_relevant_chunk_flags = []
    bm25_useless_flags = []

    for row in gold_rows:
        question_id = row["question_id"]
        question = row["question"]
        question_category = QUESTION_CATEGORY_MAP.get(question_id, "uncategorized")
        expected_sources = row["expected_sources"]
        expected_chunk_ids = row["expected_chunk_ids"]
        if len(expected_sources) == len(expected_chunk_ids):
            gold_keys = {
                make_key(source, chunk_id)
                for source, chunk_id in zip(expected_sources, expected_chunk_ids)
            }
        else:
            gold_keys = {
                make_key(source, chunk_id)
                for source in expected_sources
                for chunk_id in expected_chunk_ids
            }

        dense_docs = dense_retriever.retrieve(question, k=args.k)
        bm25_docs = bm25_retriever.retrieve(question, k=args.k)

        dense_keys = [document_to_key(doc) for doc in dense_docs]
        bm25_keys = [document_to_key(doc) for doc in bm25_docs]

        dense_key_set = set(dense_keys)
        bm25_key_set = set(bm25_keys)
        intersection = sorted(dense_key_set & bm25_key_set)
        dense_only = [key for key in dense_keys if key not in bm25_key_set]
        bm25_only = [key for key in bm25_keys if key not in dense_key_set]

        bm25_unique_relevant = [key for key in bm25_only if key in gold_keys]
        bm25_adds_new_relevant_chunk = len(bm25_unique_relevant) > 0
        bm25_is_useless = len(bm25_only) == 0 or not bm25_adds_new_relevant_chunk

        overlap_sizes.append(len(intersection))
        bm25_adds_new_relevant_chunk_flags.append(bm25_adds_new_relevant_chunk)
        bm25_useless_flags.append(bm25_is_useless)

        per_query_rows.append(
            {
                "question_id": question_id,
                "question_category": question_category,
                "question": question,
                "gold_keys": sorted(gold_keys),
                "dense_keys": dense_keys,
                "bm25_keys": bm25_keys,
                "intersection_size": len(intersection),
                "intersection_keys": intersection,
                "dense_only_keys": dense_only,
                "bm25_only_keys": bm25_only,
                "bm25_unique_relevant_keys": bm25_unique_relevant,
                "bm25_adds_new_relevant_chunk": bm25_adds_new_relevant_chunk,
                "bm25_is_useless": bm25_is_useless,
            }
        )

    summary = {
        "num_questions": len(per_query_rows),
        "k": args.k,
        "avg_overlap_at_k": round(mean(overlap_sizes), 3) if overlap_sizes else 0.0,
        "pct_queries_where_bm25_adds_new_relevant_chunk": pct_true(
            bm25_adds_new_relevant_chunk_flags
        ),
        "pct_queries_where_bm25_is_useless": pct_true(bm25_useless_flags),
        "avg_dense_unique_at_k": round(
            mean(len(row["dense_only_keys"]) for row in per_query_rows),
            3,
        )
        if per_query_rows
        else 0.0,
        "avg_bm25_unique_at_k": round(
            mean(len(row["bm25_only_keys"]) for row in per_query_rows),
            3,
        )
        if per_query_rows
        else 0.0,
        "category_summary": build_category_summary(per_query_rows),
    }

    save_jsonl(per_query_rows, args.output_jsonl)
    save_json(summary, args.summary_output)

    print(f"Saved {len(per_query_rows)} overlap rows to {args.output_jsonl}")
    print(f"Saved overlap summary to {args.summary_output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
