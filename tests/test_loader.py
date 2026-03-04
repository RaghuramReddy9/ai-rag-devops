from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents

# Load
docs = load_directory("data/raw")
print(f"\nLoaded {len(docs)} documents")

# Chunk
chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunks")

#  Inspect first 3 chunks
print("\n--- Sample Chunks ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i}:")
    print(f"  Content: {chunk.page_content[:150]}...")
    print(f"  Metadata: {chunk.metadata}")