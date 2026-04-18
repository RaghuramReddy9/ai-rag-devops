import argparse

from src.common.config import load_config
from src.eval.prediction_runner import run_predictions_with_retriever
from src.retrieval.factory import build_retriever


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
    config = load_config(config_path)
    retriever = build_retriever(config)
    run_predictions_with_retriever(
        retriever=retriever,
        output_path=config["paths"]["predictions_output"],
        config_path=config_path,
        generate_answers=False,
    )


if __name__ == "__main__":
    main()
