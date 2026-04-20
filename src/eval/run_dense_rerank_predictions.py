from __future__ import annotations

import argparse

from src.common.config import load_config
from src.eval.prediction_runner import run_prediction_flow
from src.retrieval.factory import build_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    retriever = build_retriever(config)

    run_prediction_flow(
        config=config,
        retriever=retriever,
        generate_answers=False,
    )


if __name__ == "__main__":
    main()
