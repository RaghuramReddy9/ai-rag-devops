from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.embeddings.embedder import build_vectorstore
from src.generation.generator import generate_answer, get_llm, load_prompt
from src.ingestion.chunker import chunk_documents, save_chunks_jsonl
from src.ingestion.loader import load_directory
from src.retrieval.vector_retriever import DenseRetriever


def build_index(
    raw_data_path: str,
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    chunks_output_path: str = "data/processed/chunks.jsonl",
) -> None:
    documents = load_directory(raw_data_path)
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    save_chunks_jsonl(chunks, chunks_output_path)
    build_vectorstore(
        chunks=chunks,
        persist_dir=persist_dir,
    )


def ask_question(
    question: str,
    persist_dir: str,
    k: int,
    llm,
    prompt,
) -> dict[str, Any]:
    retriever = DenseRetriever(persist_dir=persist_dir)
    retrieved_docs = retriever.retrieve(question, k=k)
    answer = generate_answer(question, retrieved_docs, llm=llm, prompt=prompt)

    citations = []
    for doc in retrieved_docs:
        metadata = doc.metadata or {}
        citations.append(
            {
                "source": metadata.get("source", "unknown"),
                "chunk_id": metadata.get("chunk_id", "unknown"),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieved_docs": retrieved_docs,
    }


if __name__ == "__main__":
    config = load_config()

    raw_data_dir = config["paths"]["raw_data_dir"]
    persist_dir = config["paths"]["persist_dir"]
    prompt_config_path = config["paths"]["prompt_config"]

    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    top_k = config["retrieval"]["top_k"]

    llm_model = config["models"]["llm_model"]

    llm = get_llm(model_name=llm_model)
    prompt = load_prompt(config_path=prompt_config_path)

    if not Path(persist_dir).exists():
        build_index(
            raw_data_path=raw_data_dir,
            persist_dir=persist_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    sample_question = "What is this dataset about?"
    result = ask_question(
        sample_question,
        persist_dir=persist_dir,
        k=top_k,
        llm=llm,
        prompt=prompt,
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nCitations:")
    for item in result["citations"]:
        print(f'- {item["source"]} | chunk_id={item["chunk_id"]}')
