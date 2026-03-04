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