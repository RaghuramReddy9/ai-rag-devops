import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return list of Documents."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {file_path}")
    return documents


def load_markdown(file_path: str) -> List[Document]:
    """Load a markdown file and return list of Documents."""
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} sections from {file_path}")
    return documents


def load_web(url: str) -> List[Document]:
    """Load a web page and return list of Documents."""
    loader = WebBaseLoader(url)
    documents = loader.load()
    print(f"Loaded {len(documents)} sections from {url}")
    return documents


def load_directory(dir_path: str) -> List[Document]:
    """
    Load all supported files from a directory.
    Supports: .pdf, .md, .txt
    Skips unsupported file types silently.
    """
    documents = []
    path = Path(dir_path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for file in path.rglob("*"):
        if file.suffix == ".pdf":
            documents.extend(load_pdf(str(file)))
        elif file.suffix in (".md", ".markdown"):
            documents.extend(load_markdown(str(file)))
        elif file.suffix == ".txt":
            # Treat plain text like markdown
            documents.extend(load_markdown(str(file)))
        else:
            # Skip silently — don't crash on .DS_Store, .gitignore etc.
            continue

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

