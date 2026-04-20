from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from src.retrieval.reranker import BaseReranker
from src.retrieval.vector_retriever import DenseRetriever


class DenseRerankRetriever:
    """Dense retrieval followed by second-stage reranking."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        reranker: BaseReranker,
        fetch_k: int = 10,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        candidate_documents = self.dense_retriever.retrieve(query=query, k=self.fetch_k)
        reranked_documents = self.reranker.rerank(
            query=query,
            documents=candidate_documents,
            top_k=k,
        )
        return [item.document for item in reranked_documents]
