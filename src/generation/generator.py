import os
import time
from types import SimpleNamespace
from typing import List
from dotenv import load_dotenv, find_dotenv

import requests
import yaml
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())


class OpenRouterChatClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def invoke(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer

        title = os.getenv("OPENROUTER_X_TITLE")
        if title:
            headers["X-Title"] = title

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return SimpleNamespace(content=content)
            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                last_error = exc
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in {429, 500, 502, 503, 504}:
                    raise
                last_error = exc
            except requests.exceptions.JSONDecodeError as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_seconds * (attempt + 1))

        raise last_error


def get_llm(model_name: str):
    """Initialize and return the LLM."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

    resolved_model = os.getenv("OPENROUTER_MODEL", model_name)
    return OpenRouterChatClient(
        model=resolved_model,
        api_key=api_key,
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
