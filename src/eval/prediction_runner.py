import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from src.common.config import load_config


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


def run_predictions_with_retriever(
    retriever,
    output_path: str,
    config_path: str,
    generate_answers: bool = False,
) -> None:
    config = load_config(config_path)
    _run_prediction_flow(
        config=config,
        retriever=retriever,
        output_path=output_path,
        generate_answers=generate_answers,
    )


def run_prediction_flow(
    config: dict,
    retriever,
    generate_answers: bool = False,
) -> None:
    output_path = config["paths"]["predictions_output"]
    _run_prediction_flow(
        config=config,
        retriever=retriever,
        output_path=output_path,
        generate_answers=generate_answers,
    )


def _run_prediction_flow(
    config: dict,
    retriever,
    output_path: str,
    generate_answers: bool = False,
) -> None:
    top_k = config["retrieval"]["top_k"]
    gold_rows = load_gold_questions(config["paths"]["gold_dataset_path"])
    max_questions = config.get("experiment", {}).get("max_questions")
    if max_questions is not None:
        gold_rows = gold_rows[:max_questions]

    llm = None
    prompt = None
    if generate_answers:
        from src.generation.generator import generate_answer, get_llm, load_prompt

        prompt_config_path = config["paths"]["prompt_config"]
        llm_model = config["models"]["llm_model"]
        llm = get_llm(model_name=llm_model)
        prompt = load_prompt(config_path=prompt_config_path)

    predictions = []

    for row in gold_rows:
        total_start = perf_counter()
        question_id = row.get("question_id")
        question = row["question"]
        retrieval_start = perf_counter()
        retrieved_docs = retriever.retrieve(question, k=top_k)
        retrieval_latency_ms = round((perf_counter() - retrieval_start) * 1000, 2)
        answer = ""
        generation_latency_ms = 0.0
        if generate_answers:
            generation_start = perf_counter()
            answer = generate_answer(question, retrieved_docs, llm=llm, prompt=prompt)
            generation_latency_ms = round((perf_counter() - generation_start) * 1000, 2)

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
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "citations": citations,
                "retrieved_context": retrieved_context,
                "latency_ms": {
                    "retrieval": retrieval_latency_ms,
                    "generation": generation_latency_ms,
                    "total": round((perf_counter() - total_start) * 1000, 2),
                },
                "run_timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    save_predictions(predictions, output_path)
    print(f"Saved {len(predictions)} predictions to {output_path}")
