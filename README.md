# AI RAG DevOps

A research-oriented Retrieval-Augmented Generation (RAG) project for comparing retrieval strategies across a shared corpus, evaluation set, and generation pipeline.

The current repository is a dense-retrieval baseline built with LangChain, Chroma, Google Gemini embeddings, and a Groq-hosted LLM. The broader goal is to use this baseline as a controlled environment for comparing:

- Dense retrieval
- BM25 / lexical retrieval
- Hybrid retrieval
- Retrieval quality under the same chunking, prompts, and evaluation setup

## Project Goal

This repo is intended as a practical research workspace rather than just a demo app.

The main idea is to keep the ingestion, chunking, prompt, and generation layers as stable as possible while swapping retrieval strategies and measuring how each one performs. That makes it easier to answer questions like:

- When does dense retrieval outperform lexical retrieval?
- Where does BM25 recover relevant chunks that embeddings miss?
- Does a hybrid strategy improve Recall@K or MRR?
- How much answer quality changes when retrieval changes but the generator stays the same?

## Current Status

What is implemented now:

- Local document ingestion from `data/raw`
- Chunking with metadata such as `chunk_id` and `chunk_size`
- Chunk export to `data/processed/chunks.jsonl`
- Dense vector indexing in Chroma
- Dense retrieval through similarity search
- Prompted answer generation using a Groq LLM
- Small gold evaluation set in `data/eval/gold_qa.jsonl`
- Prediction export to `experiments/results/baseline_predictions.jsonl`

What is planned next:

- BM25 retriever
- Hybrid retriever combining dense and lexical ranking
- A stable retrieval evaluation script for side-by-side comparison
- More datasets and broader evaluation coverage

## Repository Structure

```text
.
|-- configs/
|   |-- default.yaml
|   `-- prompts.yaml
|-- data/
|   |-- eval/
|   |   `-- gold_qa.jsonl
|   |-- processed/
|   |   `-- chunks.jsonl
|   `-- raw/
|       `-- langchain_readme.md
|-- experiments/
|   `-- results/
|       `-- baseline_predictions.jsonl
|-- src/
|   |-- common/
|   |   `-- config.py
|   |-- embeddings/
|   |   `-- embedder.py
|   |-- eval/
|   |   `-- run_predictions.py
|   |-- generation/
|   |   `-- generator.py
|   |-- ingestion/
|   |   |-- chunker.py
|   |   `-- loader.py
|   |-- retrieval/
|   |   `-- vector_retriever.py
|   `-- pipeline.py
|-- tests/
|   `-- test_pipeline.py
`-- pyproject.toml
```

## Architecture Overview

The current flow is:

1. Load source documents from `data/raw`
2. Split them into chunks with consistent metadata
3. Save the chunk corpus to `data/processed/chunks.jsonl`
4. Build a dense vector index in Chroma
5. Retrieve top-k chunks for a question
6. Format retrieved context into a prompt
7. Generate an answer with source-aware citations

See [architecture.md](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/docs/architecture.md) for a fuller design note and the planned dense vs BM25 vs hybrid comparison path.

## Core Modules

- [src/pipeline.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/pipeline.py): orchestration for indexing and question answering
- [src/ingestion/loader.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/ingestion/loader.py): loads supported local files into LangChain `Document` objects
- [src/ingestion/chunker.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/ingestion/chunker.py): chunking logic and JSONL export
- [src/embeddings/embedder.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/embeddings/embedder.py): embedding model setup and Chroma persistence
- [src/retrieval/vector_retriever.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/retrieval/vector_retriever.py): current dense retriever wrapper
- [src/generation/generator.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/generation/generator.py): prompt loading, context formatting, and generation
- [src/eval/run_predictions.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/eval/run_predictions.py): writes prediction outputs for baseline runs

## Configuration

The main runtime configuration lives in [configs/default.yaml](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/configs/default.yaml).

Current settings include:

- Raw data path
- Chroma persistence path
- Prompt config path
- Chunk size and chunk overlap
- Retrieval `top_k`
- LLM model name

Prompt templates live in [configs/prompts.yaml](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/configs/prompts.yaml).

## Data and Evaluation

The current corpus is a LangChain README-style source document:

- [data/raw/langchain_readme.md](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/data/raw/langchain_readme.md)

The current gold set is:

- [data/eval/gold_qa.jsonl](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/data/eval/gold_qa.jsonl)

Each evaluation row includes:

- `question_id`
- `question`
- `reference_answer`
- `expected_sources`
- `expected_chunk_ids`

This setup is useful for retrieval-focused research because it lets you compare which chunks each retriever returns against a known relevant target.

## Chunk Corpus Export

The chunking layer now exports a reusable chunk corpus to:

- [data/processed/chunks.jsonl](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/data/processed/chunks.jsonl)

Each row looks like:

```json
{"chunk_id": 0, "source": "data/raw/langchain_readme.md", "chunk_text": "LangChain is a framework ...", "chunk_size": 512}
```

This file is especially useful for:

- Inspecting chunk boundaries
- Debugging retrieval misses
- Reusing the same chunk corpus across dense, BM25, and hybrid retrievers
- Keeping retrieval experiments more reproducible

## Running the Baseline

Install dependencies and provide the required environment variables for the embedding model and LLM provider.

Expected environment variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (optional override for the configured model id)

Typical flow:

```bash
uv run python -m src.pipeline
```

This will:

- Load config
- Build the vector index if one does not already exist
- Save the chunk corpus JSONL
- Run a sample question through the dense baseline

To generate saved prediction outputs:

```bash
uv run python -m src.eval.run_predictions
```

## Research Directions

This repo is a good fit for the next set of comparisons:

- Dense vs BM25 on the same chunk corpus
- Hybrid fusion strategies such as score fusion or reciprocal rank fusion
- Retrieval metrics such as Recall@1, Recall@3, Recall@5, and MRR
- Retrieval error analysis using the saved chunk corpus and prediction outputs
- Prompt sensitivity after retrieval quality changes

## Current Limitations

- The corpus is still very small
- The current checked-in retrieval path is dense only
- The evaluation set is useful but narrow
- The test coverage is still light
- The project is still closer to a research baseline than a production-ready system

## Next Improvements

- Add a BM25 retriever implementation over `chunks.jsonl`
- Add a hybrid retriever with a configurable fusion strategy
- Restore or extend retrieval evaluation scripts
- Expand the corpus and gold labels
- Add cleaner experiment tracking and result summaries
- Add real automated tests for retrieval and evaluation behavior

## Notes

This repository is intentionally evolving in a research direction. The priority is not only to generate answers, but to understand retrieval behavior, compare strategies fairly, and build a repeatable evaluation workflow for RAG experimentation.
