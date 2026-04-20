from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document


@dataclass
class RerankedDocument:
    document: Document
    score: float


class BaseReranker(ABC):
    """Interface for reranking a retrieved candidate set for a query."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[RerankedDocument]:
        """Return reranked documents with scores for the given query."""


class IdentityReranker(BaseReranker):
    """Pass-through reranker useful as a baseline before model integration."""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[RerankedDocument]:
        del query

        capped_documents = documents[:top_k] if top_k is not None else documents
        return [
            RerankedDocument(document=document, score=float(len(capped_documents) - rank))
            for rank, document in enumerate(capped_documents)
        ]
