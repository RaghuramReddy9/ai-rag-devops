import argparse
from pathlib import Path

from src.common.config import load_config
from src.embeddings.embedder import vectorstore_has_documents
from src.eval.prediction_runner import run_predictions_with_retriever
from src.pipeline import build_index
from src.retrieval.factory import build_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval predictions.")
    parser.add_argument(
        "--config",
        default="configs/hybrid.yaml",
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    raw_data_dir = config["paths"]["raw_data_dir"]
    persist_dir = config["paths"]["persist_dir"]
    chunks_output = config["paths"]["chunks_output"]
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    if not Path(persist_dir).exists() or not vectorstore_has_documents(persist_dir):
        build_index(
            raw_data_path=raw_data_dir,
            persist_dir=persist_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunks_output_path=chunks_output,
        )

    retriever = build_retriever(config)
    run_predictions_with_retriever(
        retriever=retriever,
        output_path=config["paths"]["predictions_output"],
        config_path=config_path,
        generate_answers=False,
    )


if __name__ == "__main__":
    main()
