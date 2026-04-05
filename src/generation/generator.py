import os
from typing import List
from dotenv import load_dotenv, find_dotenv

import yaml
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())


def get_llm(model_name: str):
    """Initialize and return the LLM."""
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )


def load_prompt(
    prompt_key: str = "rag_prompt",
    config_path: str = "configs/prompts.yaml",
) -> PromptTemplate:
    """Load prompt template from version-controlled config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    template = prompts[prompt_key]
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


def format_context(chunks: List[Document]) -> str:
    context_parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", "?")
        context_parts.append(
            f"[Source: {source}, Chunk ID: {chunk_id}]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    question: str,
    chunks: List[Document],
    llm,
    prompt: PromptTemplate,
) -> str:
    context = format_context(chunks)
    formatted_prompt = prompt.format(context=context, question=question)

    response = llm.invoke(formatted_prompt)
    return response.content.strip()