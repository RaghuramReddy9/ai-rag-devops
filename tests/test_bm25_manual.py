from src.retrieval.bm25_retriever import BM25Retriever


retriever = BM25Retriever()
docs = retriever.retrieve("What is LangChain?", k=3)

for doc in docs:
    print(doc.metadata)
