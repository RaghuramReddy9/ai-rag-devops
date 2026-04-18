from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.retrieval.vector_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever


FusionKey = Tuple[str, int]


@dataclass
class RankedCandidate:
    key: FusionKey
    document: Document
    rrf_score: float


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        dense_fetch_k: int | None = None,
        bm25_fetch_k: int | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_fetch_k = dense_fetch_k
        self.bm25_fetch_k = bm25_fetch_k

    def _make_fusion_key(self, document: Document) -> FusionKey:
        metadata = document.metadata or {}

        if "source" not in metadata:
            raise ValueError("Document metadata missing required field: 'source'")

        if "chunk_id" not in metadata:
            raise ValueError("Document metadata missing required field: 'chunk_id'")

        source = str(metadata["source"]).replace("\\", "/")

        try:
            chunk_id = int(metadata["chunk_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Document metadata field 'chunk_id' must be int-compatible, got: {metadata['chunk_id']!r}"
            ) from exc

        return (source, chunk_id)

    def _accumulate_rrf_scores(
        self,
        documents: List[Document],
        fused_scores: Dict[FusionKey, float],
        fused_documents: Dict[FusionKey, Document],
    ) -> None:
        for rank, document in enumerate(documents, start=1):
            key = self._make_fusion_key(document)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))
            fused_documents[key] = document

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        dense_k = self.dense_fetch_k if self.dense_fetch_k is not None else k
        bm25_k = self.bm25_fetch_k if self.bm25_fetch_k is not None else k

        dense_docs = self.dense_retriever.retrieve(query=query, k=dense_k)
        bm25_docs = self.bm25_retriever.retrieve(query=query, k=bm25_k)

        fused_scores: Dict[FusionKey, float] = {}
        fused_documents: Dict[FusionKey, Document] = {}

        self._accumulate_rrf_scores(
            documents=dense_docs,
            fused_scores=fused_scores,
            fused_documents=fused_documents,
        )
        self._accumulate_rrf_scores(
            documents=bm25_docs,
            fused_scores=fused_scores,
            fused_documents=fused_documents,
        )

        ranked_candidates = sorted(
            (
                RankedCandidate(
                    key=key,
                    document=fused_documents[key],
                    rrf_score=score,
                )
                for key, score in fused_scores.items()
            ),
            key=lambda candidate: candidate.rrf_score,
            reverse=True,
        )

        return [candidate.document for candidate in ranked_candidates[:k]]
