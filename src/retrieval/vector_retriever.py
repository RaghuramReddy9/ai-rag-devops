from typing import List

from langchain_core.documents import Document

from src.embeddings.embedder import get_embeddings, load_vectorstore


class DenseRetriever:
    def __init__(self, persist_dir: str = "chroma_db") -> None:
        self.persist_dir = persist_dir
        self.vector_store = load_vectorstore(persist_dir=self.persist_dir)
        

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)