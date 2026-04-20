# Architecture

## Purpose

This repository has two stable layers:

- serving-oriented retrieval and answer generation
- research-oriented benchmarking and diagnostics

The current serving stack is:

```text
Dense retrieval
    ->
Cross-encoder reranking
    ->
Final top-k context
    ->
Prompted answer generation
```

That is the current architecture, not future work.

## System Modes

### Retrieval-Only Mode

Used for:
- dense
- hybrid
- dense_rerank
- BM25 diagnostics

Flow:

```text
Raw documents
    ->
Chunking + metadata
    ->
chunks.jsonl
    ->
Retriever
    ->
Retrieved documents
    ->
Evaluation
```

No LLM call happens in this mode.

This mode exists to answer:
- does the retriever find the right chunks?
- does reranking improve ranking quality?
- does hybrid help enough to justify its complexity?

### Answer-Generation Mode

Used for:
- dense + LLM
- dense_rerank + LLM

Flow:

```text
Retriever
    ->
Retrieved documents
    ->
Prompt assembly
    ->
LLM
    ->
Answer + citations + latency
```

This mode exists to answer:
- do retrieval gains improve final answers?
- are answers grounded in retrieved evidence?
- is the extra latency worth it?

## Current Serving Stack

### 1. Ingestion

`src/ingestion/loader.py`

Responsibilities:
- load files from `data/raw`
- produce LangChain `Document` objects

### 2. Chunking

`src/ingestion/chunker.py`

Responsibilities:
- split documents into chunks
- attach `chunk_id`
- attach `chunk_size`
- export `data/processed/chunks.jsonl`

### 3. Dense Indexing

`src/embeddings/embedder.py`

Responsibilities:
- build MiniLM embeddings locally
- persist Chroma
- reload the vector store

### 4. Retrieval

Base retriever:
- `src/retrieval/vector_retriever.py`

Reranker abstraction:
- `src/retrieval/reranker.py`

Cross-encoder implementation:
- `src/retrieval/cross_encoder_reranker.py`

Dense + rerank wrapper:
- `src/retrieval/dense_rerank_retriever.py`

Factory:
- `src/retrieval/factory.py`

Current serving retrieval path:

```text
DenseRetriever(fetch_k)
    ->
CrossEncoderReranker
    ->
final top_k documents
```

### 5. Generation

`src/generation/generator.py`

Responsibilities:
- build the LLM client
- load the prompt
- format retrieved context
- generate the answer

### 6. Prediction Flow

`src/eval/prediction_runner.py`

Responsibilities:
- load the gold question set
- run retrieval or retrieval+generation
- save predictions in one shared schema
- capture latency

This shared flow is important because it lets experiments stay identical except for the retriever.

## Research Artifacts

### BM25

`src/retrieval/bm25_retriever.py`

Retained for:
- lexical baseline
- overlap analysis
- research comparison

Not used as serving default because:
- overlap analysis showed little useful additional relevance
- answer benchmark evidence did not justify making it part of the serving path

### Hybrid

`src/retrieval/hybrid_retriever.py`

Retained for:
- fusion experiments
- ranking analysis

Not used as serving default because:
- it improved early ranking but not enough broader coverage
- dense+rerank produced a stronger overall stack

## Evaluation Layers

### Retrieval Metrics

Used metrics:
- `MRR`
- `Recall@1`
- `Recall@3`
- `Recall@5`
- overlap diagnostics

### Answer Metrics

Current lightweight scorecard:
- correctness proxy
- expected chunk retrieved rate
- citation grounding
- unsupported-risk proxy
- latency:
  - retrieval
  - generation
  - total

## Why Dense + Rerank Won

The current architecture settles on dense+rereank because:

- dense provides strong candidate recall
- the cross-encoder improves ranking inside that candidate pool
- answer quality improved
- unsupported-risk dropped
- citation grounding stayed strong

## Practical Summary

This repo should now be understood as:

- dense+rerank is the serving retrieval stack
- BM25 and hybrid are research artifacts
- retrieval-only benchmarking and answer-generation benchmarking are intentionally separate modes
