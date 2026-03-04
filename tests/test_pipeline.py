import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import build_vectorstore, load_vectorstore

# Step 1: Load
docs = load_directory("data/raw")

# Step 2: Chunk
chunks = chunk_documents(docs)

# Step 3: Embed and store
vectorstore = build_vectorstore(chunks)

# Step 4: Test a simple search
print("\n--- Testing Semantic Search ---")
query = "how do I install LangChain?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Content: {doc.page_content[:200]}...")
    print(f"  Source: {doc.metadata.get('source')}")
    print(f"  Chunk ID: {doc.metadata.get('chunk_id')}")