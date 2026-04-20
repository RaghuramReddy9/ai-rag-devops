import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.embedder import build_vectorstore, load_vectorstore
from src.generation.generator import generate_answer
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_directory


if os.path.exists("chroma_db"):
    print("Loading existing vector store...")
    vectorstore = load_vectorstore()
else:
    print("Building vector store from scratch...")
    docs = load_directory("data/raw")
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)

questions = [
    "How do I install LangChain?",
    "What is LangGraph used for?",
    "What models does LangChain support?",
]

print("\n" + "=" * 60)
print("RAG PIPELINE SMOKE SCRIPT")
print("=" * 60)

for question in questions:
    print(f"\nQuestion: {question}")
    print("-" * 40)

    chunks = vectorstore.similarity_search(question, k=3)
    answer, sources = generate_answer(question, chunks)

    print(f"Answer: {answer}")
    print(f"\nSources used: {[c.metadata.get('chunk_id') for c in sources]}")
    print("=" * 60)
