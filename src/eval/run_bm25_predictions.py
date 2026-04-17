from src.common.config import load_config
from src.eval.prediction_runner import run_predictions_with_retriever
from src.retrieval.bm25_retriever import BM25Retriever


def main() -> None:
    config_path = "configs/bm25.yaml"
    retriever = BM25Retriever()
    run_predictions_with_retriever(
        retriever=retriever,
        output_path=load_config(config_path)["paths"]["predictions_output"],
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
