import json
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Split documents into chunks for embedding.
    
    Args:
        documents: List of Document objects from loader
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunked Document objects with metadata preserved
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata so we can trace which chunk answered the question
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    return chunks


def save_chunks_jsonl(chunks: List[Document], output_path: str) -> None:
    """Persist chunk metadata and text as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            metadata = chunk.metadata or {}
            source = str(metadata.get("source", "unknown")).replace("\\", "/")
            row = {
                "chunk_id": metadata.get("chunk_id", "unknown"),
                "source": source,
                "chunk_text": chunk.page_content,
                "chunk_size": metadata.get("chunk_size", len(chunk.page_content)),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {output_path}")
