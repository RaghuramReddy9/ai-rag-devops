import json
from datetime import datetime
from pathlib import Path

from src.common.config import load_config
from src.generation.generator import get_llm, load_prompt
from src.pipeline import ask_question


def save_predictions(predictions: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in predictions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    config = load_config()

    persist_dir = config["paths"]["persist_dir"]
    prompt_config_path = config["paths"]["prompt_config"]
    top_k = config["retrieval"]["top_k"]
    llm_model = config["models"]["llm_model"]

    llm = get_llm(model_name=llm_model)
    prompt = load_prompt(config_path=prompt_config_path)

    questions = [
        "What is this dataset about?",
        "What can LangChain be used for?",
        "What does LangSmith help with?",
    ]

    predictions = []

    for question in questions:
        result = ask_question(
            question=question,
            persist_dir=persist_dir,
            k=top_k,
            llm=llm,
            prompt=prompt,
        )

        retrieved_context = []
        for doc in result["retrieved_docs"]:
            metadata = doc.metadata or {}
            retrieved_context.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "chunk_id": metadata.get("chunk_id", "unknown"),
                    "chunk_text": doc.page_content,
                }
            )

        predictions.append(
            {
                "question": result["question"],
                "answer": result["answer"],
                "citations": result["citations"],
                "retrieved_context": retrieved_context,
                "run_timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    output_path = "experiments/results/baseline_predictions.jsonl"
    save_predictions(predictions, output_path)

    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()