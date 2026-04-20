from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from src.retrieval.reranker import BaseReranker, RerankedDocument


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker backed by a lightweight Hugging Face model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size: int = 16,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[RerankedDocument]:
        if not documents:
            return []

        pairs = [(query, document.page_content) for document in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked_documents = sorted(
            (
                RerankedDocument(
                    document=document,
                    score=float(score),
                )
                for document, score in zip(documents, scores)
            ),
            key=lambda item: item.score,
            reverse=True,
        )

        return ranked_documents[:top_k] if top_k is not None else ranked_documents
