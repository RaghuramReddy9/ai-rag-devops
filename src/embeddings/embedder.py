import os
from typing import List
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv(find_dotenv())


def get_embeddings():
    """Initialize and return the embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def build_vectorstore(chunks: List[Document], persist_dir: str = "chroma_db") -> Chroma:
    """
    Embed chunks and store them in ChromaDB.
    
    Args:
        chunks: List of chunked Document objects
        persist_dir: Directory to persist the vector store
    
    Returns:
        Chroma vectorstore instance
    """
    embeddings = get_embeddings()

    print(f"Building vector store from {len(chunks)} chunks...")
    print("This may take a moment — embedding each chunk via Gemini API...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Vector store built and persisted to '{persist_dir}/'")
    print(f"Total vectors stored: {vectorstore._collection.count()}")

    return vectorstore


def load_vectorstore(persist_dir: str = "chroma_db") -> Chroma:
    """
    Load an existing vector store from disk.
    Use this after the first run — no need to re-embed every time.
    
    Args:
        persist_dir: Directory where vector store is persisted
    
    Returns:
        Chroma vectorstore instance
    """
    embeddings = get_embeddings()

    print(f"Loading existing vector store from '{persist_dir}/'...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    print(f"Loaded vector store with {vectorstore._collection.count()} vectors")
    return vectorstore