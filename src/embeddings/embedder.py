from pathlib import Path
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.common.config import load_config

load_dotenv(find_dotenv())


def get_embeddings():
    """Initialize and return the local sentence-transformer embedding model."""
    config = load_config()
    model_name = config["models"]["embedding_model"]

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document], persist_dir: str = "chroma_db_local") -> Chroma:
    """
    Embed chunks and store them in ChromaDB.

    Args:
        chunks: List of chunked Document objects
        persist_dir: Directory to persist the vector store

    Returns:
        Chroma vectorstore instance
    """
    config = load_config()
    model_name = config["models"]["embedding_model"]
    embeddings = get_embeddings()

    print(f"Building vector store from {len(chunks)} chunks...")
    print(f"Using local sentence-transformer model: {model_name}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Vector store built and persisted to '{persist_dir}/'")
    print(f"Total vectors stored: {vectorstore._collection.count()}")

    return vectorstore


def load_vectorstore(persist_dir: str = "chroma_db_local") -> Chroma:
    """
    Load an existing vector store from disk.
    Use this after the first run, with no API embedding calls required.

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


def vectorstore_has_documents(persist_dir: str = "chroma_db_local") -> bool:
    """Return True when the persisted Chroma store exists and contains vectors."""
    if not Path(persist_dir).exists():
        return False

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )
    return vectorstore._collection.count() > 0
