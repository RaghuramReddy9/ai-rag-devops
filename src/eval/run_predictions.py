from src.common.config import load_config
from src.eval.prediction_runner import run_predictions_with_retriever
from src.retrieval.vector_retriever import DenseRetriever


def main() -> None:
    config_path = "configs/dense.yaml"
    config = load_config(config_path)
    persist_dir = config["paths"]["persist_dir"]
    retriever = DenseRetriever(persist_dir=persist_dir)
    run_predictions_with_retriever(
        retriever=retriever,
        output_path=config["paths"]["predictions_output"],
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
