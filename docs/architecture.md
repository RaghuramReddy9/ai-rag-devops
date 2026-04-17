# Architecture

## Purpose

This project is structured as a research baseline for retrieval comparison in RAG systems.

The architecture is designed so that ingestion, chunking, corpus export, prompting, and generation can remain mostly stable while retrieval strategies are swapped and evaluated. That gives you a cleaner way to study the effect of retrieval on final performance.

## Current System

The currently implemented system is a dense-retrieval RAG pipeline.

```text
Raw documents
    ->
Document loaders
    ->
Chunking + metadata
    ->
chunks.jsonl export
    ->
Dense embeddings
    ->
Chroma vector index
    ->
Dense retrieval
    ->
Prompt assembly
    ->
LLM answer generation
```

## Components

### 1. Ingestion

[loader.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/ingestion/loader.py) loads local documents into LangChain `Document` objects.

Supported sources currently include:

- PDF
- Markdown
- Plain text
- Web pages through a loader function

For the current repo state, the active corpus is a Markdown file in `data/raw`.

### 2. Chunking

[chunker.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/ingestion/chunker.py) splits documents using `RecursiveCharacterTextSplitter`.

Each chunk is annotated with:

- `chunk_id`
- `chunk_size`
- source metadata inherited from the loader

The same module also exports the chunk corpus to `data/processed/chunks.jsonl`, which is an important research artifact because it gives all retrievers the same text units.

### 3. Dense Indexing

[embedder.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/embeddings/embedder.py) creates embeddings with Google Gemini and persists them in Chroma.

This layer is responsible for:

- Embedding chunk text
- Building the vector store
- Reloading a persisted vector store for future runs

### 4. Retrieval

[vector_retriever.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/retrieval/vector_retriever.py) is the current retriever implementation.

It performs:

- Dense similarity search
- Top-k chunk retrieval

This is the main baseline that future retrieval variants should be compared against.

### 5. Prompting and Generation

[generator.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/generation/generator.py) handles:

- LLM client setup
- Prompt loading from YAML
- Retrieved-context formatting
- Final answer generation

The prompt is intentionally strict about using only retrieved context and including source citations.

### 6. Pipeline Orchestration

[pipeline.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/pipeline.py) ties the pieces together.

It currently supports:

- Building the dense index
- Saving the chunk corpus
- Running retrieval and generation for a question

## Research Architecture Direction

The next step is to evolve the architecture from a single dense retriever into a retrieval comparison framework.

Target direction:

```text
Shared raw documents
    ->
Shared chunking pipeline
    ->
Shared chunks.jsonl corpus
    ->
Multiple retrievers
    |-> Dense
    |-> BM25
    |-> Hybrid
    ->
Common retrieval interface
    ->
Common evaluation harness
    ->
Experiment outputs and comparison summaries
```

## Recommended Retrieval Abstraction

To support comparison cleanly, retrieval should move toward a common interface such as:

```python
class BaseRetriever:
    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        ...
```

Then each retriever can implement the same contract:

- `DenseRetriever`
- `BM25Retriever`
- `HybridRetriever`

That keeps the rest of the system unchanged and makes experiments easier to compare.

## Planned Retrieval Modes

### Dense Retrieval

Best for:

- Semantic matching
- Paraphrased questions
- Queries where lexical overlap is weak

Tradeoff:

- Can miss exact-keyword matches or domain-specific terms

### BM25 Retrieval

Best for:

- Exact phrase or keyword matching
- Sparse, term-heavy queries
- Situations where important words must be matched directly

Tradeoff:

- Can miss semantically similar but lexically different passages

### Hybrid Retrieval

Best for:

- Balancing semantic recall and keyword precision
- Reducing failure modes of either retrieval strategy alone

Possible fusion strategies:

- Weighted score fusion
- Reciprocal rank fusion
- Two-stage retrieval and reranking

## Evaluation View

The architecture should support at least two levels of evaluation.

### Retrieval Evaluation

Goal:

- Measure whether the retriever finds the correct chunks

Useful metrics:

- Recall@K
- MRR
- Hit rate
- Rank of first relevant chunk

Artifacts:

- Question-level retrieval outputs
- Aggregate summary JSON

### End-to-End RAG Evaluation

Goal:

- Measure whether better retrieval leads to better answers

Useful checks:

- Citation correctness
- Groundedness to retrieved context
- Coverage of the expected answer
- Comparison across retriever types using the same prompt and model

## Why `chunks.jsonl` Matters

The chunk export is important architecturally, not just operationally.

It gives the system:

- A stable shared corpus representation
- Easier debugging of chunk boundaries
- A common input for BM25 indexing
- Better reproducibility across experiments
- A simpler way to inspect retrieval failures without reopening source documents every time

## Suggested Next Refactor

If you continue in the research direction, the cleanest next architecture update would be:

1. Add a `BM25Retriever` built from `data/processed/chunks.jsonl`
2. Add a `HybridRetriever` using a simple fusion strategy
3. Add a retriever selector in config
4. Add a retrieval evaluation script that saves per-retriever outputs in separate experiment folders
5. Keep the same chunking and prompting path so comparisons stay fair

## Practical Summary

Right now the repo is a dense baseline with research scaffolding already visible.

The architecture is in a good place to become a retrieval comparison framework because:

- chunking is already separated
- chunk metadata is preserved
- chunk corpus export now exists
- retrieval is isolated behind a small wrapper
- generation is independent from retrieval implementation details

That means the project can grow from "a working dense RAG prototype" into "a repeatable research repo for dense vs BM25 vs hybrid retrieval experiments" without needing a full redesign.
