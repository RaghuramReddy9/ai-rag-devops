from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import DenseRetriever


def build_retriever(config: dict):
    retrieval_type = config["retrieval"]["type"]

    if retrieval_type == "dense":
        return DenseRetriever(persist_dir=config["paths"]["persist_dir"])

    if retrieval_type == "bm25":
        return BM25Retriever(chunks_path=config["paths"]["chunks_output"])

    if retrieval_type == "hybrid":
        dense_retriever = DenseRetriever(persist_dir=config["paths"]["persist_dir"])
        bm25_retriever = BM25Retriever(chunks_path=config["paths"]["chunks_output"])
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=config["retrieval"].get("rrf_k", 60),
            dense_fetch_k=config["retrieval"].get("dense_fetch_k"),
            bm25_fetch_k=config["retrieval"].get("bm25_fetch_k"),
        )

    raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
