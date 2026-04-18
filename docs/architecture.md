# Architecture

## Purpose

This repository has two layers:

- main application pipeline
- retrieval research bench

Those two layers should not be confused.

Main application pipeline:
- dense retrieval over Chroma

Research bench:
- dense
- BM25
- hybrid

That split is intentional and reflects the current benchmark results.

## Main Serving Path

The serving path is dense-first.

```text
Raw documents
    ->
Document loaders
    ->
Chunking + metadata
    ->
chunks.jsonl export
    ->
MiniLM embeddings
    ->
Chroma vector index
    ->
Dense retrieval
    ->
Prompt assembly
    ->
LLM answer generation
    ->
Citations
```

Why dense is the serving path:
- best retrieval coverage on the current benchmark
- BM25 contributes little useful additional relevance
- hybrid helps rank-first results but does not beat dense coverage at `@3` and `@5`

## Components

### Ingestion

[loader.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/ingestion/loader.py)

Responsibilities:
- load raw files from `data/raw`
- return LangChain `Document` objects

### Chunking

[chunker.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/ingestion/chunker.py)

Responsibilities:
- split documents into chunks
- assign `chunk_id`
- assign `chunk_size`
- export [chunks.jsonl](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/data/processed/chunks.jsonl)

Why `chunks.jsonl` still matters:
- shared artifact for debugging
- shared artifact for BM25 benchmarking
- stable view of the corpus for evaluation and inspection

### Dense Indexing

[embedder.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/embeddings/embedder.py)

Responsibilities:
- create local MiniLM embeddings
- build Chroma
- reload Chroma for retrieval

Important rule:
- one vector store must use one embedding model consistently

### Dense Retrieval

[vector_retriever.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/retrieval/vector_retriever.py)

Responsibilities:
- query Chroma
- return top-k dense results

This is the primary retriever used by [pipeline.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/pipeline.py).

### Generation

[generator.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/generation/generator.py)

Responsibilities:
- initialize the LLM client
- load the prompt
- format retrieved context
- generate the final answer

### Pipeline Orchestration

[pipeline.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/pipeline.py)

Responsibilities:
- build the dense index if needed
- save chunks
- run dense retrieval
- call generation
- attach citations

This file is the main application path and should stay dense-only unless benchmark evidence clearly changes.

## Research Bench

The repo still keeps BM25 and hybrid, but as experiment artifacts rather than serving defaults.

Artifacts:
- [bm25_retriever.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/retrieval/bm25_retriever.py)
- [hybrid_retriever.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/retrieval/hybrid_retriever.py)
- [factory.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/retrieval/factory.py)

Experiment runners:
- [run_predictions.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/eval/run_predictions.py)
- [run_bm25_predictions.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/eval/run_bm25_predictions.py)
- [run_hybrid_predictions.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/eval/run_hybrid_predictions.py)

Experiment evaluators:
- [retrieval_eval.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/eval/retrieval_eval.py)
- [analyze_retrieval_overlap.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/eval/analyze_retrieval_overlap.py)

### Why BM25 Is Not in the Main Path

Overlap analysis currently shows:
- `avg_overlap@5 = 2.104`
- BM25 adds a new relevant chunk on `6.2%` of queries
- BM25 is useless on `93.8%` of queries

That means BM25 remains useful as:
- a baseline
- a diagnostic tool
- a future routing signal

But not as:
- the main retriever for the serving path

### Why Hybrid Is Still Kept

Hybrid is still useful because:
- it improved `MRR`
- it improved `Recall@1`
- it is a valid experiment artifact for ranking analysis

But hybrid is not the main serving path because:
- dense still has stronger retrieval coverage at `Recall@3` and `Recall@5`
- deeper branch fetch did not improve the hybrid summary on the current benchmark

## Evaluation Layers

The architecture now assumes evaluation should happen in stages.

### 1. Retrieval-Only Evaluation

Goal:
- measure whether the retriever finds the right chunks

Metrics:
- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`
- overlap and diversity diagnostics

This stage should run without the LLM in the loop.

### 2. Reranking

Goal:
- improve ordering after dense retrieval

Recommended next layer:
- cross-encoder reranking on the dense candidate set

This is the next quality layer to add.

### 3. Grounding and Citation Enforcement

Goal:
- make sure generated answers stay attached to retrieved evidence

This stage should check:
- citation presence
- citation correctness
- answer grounding to retrieved chunks

### 4. End-to-End Answer Evaluation

Goal:
- judge final answer usefulness only after retrieval and grounding are stable

## Recommended Next Direction

The next architecture change should be:

1. dense retrieval
2. cross-encoder reranking
3. grounding and citation enforcement
4. end-to-end answer evaluation

Not the other way around.

## Practical Summary

This repo is no longer best described as “dense vs BM25 vs hybrid as equal pipeline options.”

It is better described as:

- dense is the primary application retriever
- BM25 and hybrid are controlled research artifacts
- reranking is the next system improvement
- grounding is the next answer-safety improvement

See also:
- [findings.md](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/docs/findings.md)
