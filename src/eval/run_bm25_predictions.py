import argparse

from src.common.config import load_config
from src.eval.prediction_runner import run_predictions_with_retriever
from src.retrieval.bm25_retriever import BM25Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval predictions.")
    parser.add_argument(
        "--config",
        default="configs/bm25.yaml",
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    retriever = BM25Retriever()
    run_predictions_with_retriever(
        retriever=retriever,
        output_path=load_config(config_path)["paths"]["predictions_output"],
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
