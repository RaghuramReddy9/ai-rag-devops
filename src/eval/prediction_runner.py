import json
from datetime import datetime
from pathlib import Path

from src.common.config import load_config
from src.generation.generator import generate_answer, get_llm, load_prompt


def load_gold_questions(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_predictions(predictions: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in predictions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_predictions_with_retriever(retriever, output_path: str, config_path: str) -> None:
    config = load_config(config_path)

    prompt_config_path = config["paths"]["prompt_config"]
    top_k = config["retrieval"]["top_k"]
    llm_model = config["models"]["llm_model"]

    llm = get_llm(model_name=llm_model)
    prompt = load_prompt(config_path=prompt_config_path)
    gold_rows = load_gold_questions(config["paths"]["gold_dataset_path"])

    predictions = []

    for row in gold_rows:
        question = row["question"]
        retrieved_docs = retriever.retrieve(question, k=top_k)
        answer = generate_answer(question, retrieved_docs, llm=llm, prompt=prompt)

        citations = []
        retrieved_context = []
        for doc in retrieved_docs:
            metadata = doc.metadata or {}
            citations.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "chunk_id": metadata.get("chunk_id", "unknown"),
                }
            )
            retrieved_context.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "chunk_id": metadata.get("chunk_id", "unknown"),
                    "chunk_text": doc.page_content,
                }
            )

        predictions.append(
            {
                "question": question,
                "answer": answer,
                "citations": citations,
                "retrieved_context": retrieved_context,
                "run_timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    save_predictions(predictions, output_path)
    print(f"Saved {len(predictions)} predictions to {output_path}")
