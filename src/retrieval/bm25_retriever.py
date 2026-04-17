import json
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks_path: str = "data/processed/chunks.jsonl") -> None:
        self.chunks_path = Path(chunks_path)
        self.documents = self._load_documents()
        self.tokenized_corpus = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_documents(self) -> list[Document]:
        documents = []

        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                documents.append(
                    Document(
                        page_content=row["chunk_text"],
                        metadata={
                            "chunk_id": row["chunk_id"],
                            "source": row["source"],
                            "chunk_size": row.get("chunk_size"),
                        },
                    )
                )

        return documents

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [self.documents[i] for i in ranked_indices]
