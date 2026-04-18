# AI RAG DevOps

A research-oriented RAG repository centered on one primary retrieval path and a separate experimental bench.

Main pipeline:
- Dense retrieval over Chroma using local `sentence-transformers/all-MiniLM-L6-v2`

Experiment artifacts:
- BM25 retrieval over `data/processed/chunks.jsonl`
- Hybrid retrieval with reciprocal rank fusion

The repo is now intentionally moving in this order:
1. Retrieval quality
2. Reranking
3. Grounding and citation enforcement
4. End-to-end answer generation

## Main Decision

Dense retrieval is the primary retriever for the main pipeline.

Why:
- Dense is currently the strongest retriever on the gold benchmark
- BM25 adds a new relevant chunk on only a small fraction of queries
- Hybrid is useful as a benchmark artifact, but it is not the main application path

So the repo should be read as:
- main application path = dense
- research comparison path = dense vs BM25 vs hybrid

## Current Retrieval Results

On the current 48-question gold set:

- Dense
  - `MRR: 0.674`
  - `Recall@1: 0.486`
  - `Recall@3: 0.847`
  - `Recall@5: 0.847`
- BM25
  - `MRR: 0.496`
  - `Recall@1: 0.392`
  - `Recall@3: 0.576`
  - `Recall@5: 0.576`
- Hybrid
  - `MRR: 0.705`
  - `Recall@1: 0.59`
  - `Recall@3: 0.753`
  - `Recall@5: 0.753`

Interpretation:
- Hybrid improves first-hit ranking
- Dense still has the best coverage at `@3` and `@5`
- BM25 is mainly useful as a comparison baseline, not as the main retrieval path

## Repository Structure

```text
.
|-- configs/
|   |-- default.yaml
|   |-- dense.yaml
|   |-- bm25.yaml
|   |-- hybrid.yaml
|   `-- prompts.yaml
|-- data/
|   |-- eval/
|   |   `-- gold_qa.jsonl
|   |-- processed/
|   |   `-- chunks.jsonl
|   `-- raw/
|-- docs/
|   `-- architecture.md
|-- experiments/
|   `-- results/
|-- src/
|   |-- common/
|   |   `-- config.py
|   |-- embeddings/
|   |   `-- embedder.py
|   |-- eval/
|   |   |-- analyze_retrieval_overlap.py
|   |   |-- prediction_runner.py
|   |   |-- retrieval_eval.py
|   |   |-- run_predictions.py
|   |   |-- run_bm25_predictions.py
|   |   `-- run_hybrid_predictions.py
|   |-- generation/
|   |   `-- generator.py
|   |-- ingestion/
|   |   |-- chunker.py
|   |   `-- loader.py
|   |-- retrieval/
|   |   |-- bm25_retriever.py
|   |   |-- factory.py
|   |   |-- hybrid_retriever.py
|   |   `-- vector_retriever.py
|   `-- pipeline.py
|-- tests/
`-- pyproject.toml
```

## Main Pipeline

The main runtime path is dense-only.

Flow:
1. Load raw documents from `data/raw`
2. Chunk them with metadata
3. Save `data/processed/chunks.jsonl`
4. Build Chroma with local MiniLM embeddings
5. Retrieve with `DenseRetriever`
6. Pass retrieved context to the LLM
7. Return answer plus citations

Main orchestration:
- [pipeline.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/pipeline.py)

Core dense components:
- [embedder.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/embeddings/embedder.py)
- [vector_retriever.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/retrieval/vector_retriever.py)
- [generator.py](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/src/generation/generator.py)

## Experimental Retrieval Bench

BM25 and hybrid are still kept in the repo, but as research artifacts.

They are useful for:
- side-by-side retrieval comparison
- overlap analysis
- future routing experiments
- documenting why dense is the main retriever

Experiment configs:
- [dense.yaml](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/configs/dense.yaml)
- [bm25.yaml](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/configs/bm25.yaml)
- [hybrid.yaml](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/configs/hybrid.yaml)

## Retrieval-Only Evaluation

The retrieval comparison runs without putting the LLM in the loop.

That is intentional.

This repo now treats retrieval benchmarking as a separate stage from answer generation.

Research notes:
- [architecture.md](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/docs/architecture.md)
- [findings.md](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/docs/findings.md)

Useful commands:

```bash
uv run python -m src.eval.run_predictions --config configs/dense.yaml
uv run python -m src.eval.run_bm25_predictions --config configs/bm25.yaml
uv run python -m src.eval.run_hybrid_predictions --config configs/hybrid.yaml

uv run python -m src.eval.retrieval_eval --config configs/dense.yaml
uv run python -m src.eval.retrieval_eval --config configs/bm25.yaml
uv run python -m src.eval.retrieval_eval --config configs/hybrid.yaml

uv run python -m src.eval.analyze_retrieval_overlap --k 5
```

## Overlap Analysis

The current overlap diagnostic shows:

- `avg_overlap@5 = 2.104`
- `BM25 adds a new relevant chunk on 6.2% of queries`
- `BM25 is useless on 93.8% of queries`

This is the main reason BM25 is no longer part of the main application path.

Artifacts:
- [retrieval_overlap_summary.json](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/experiments/results/retrieval_overlap_summary.json)
- [retrieval_overlap_analysis.jsonl](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/experiments/results/retrieval_overlap_analysis.jsonl)

Decision summary:
- [findings.md](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/docs/findings.md)

## Environment

Expected environment variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` optional override

Dense embeddings are local and do not require a remote embedding API key.

## Next Work

The next engineering steps are:

1. Cross-encoder reranking on top of dense retrieval
2. Grounding and citation enforcement
3. Then end-to-end answer evaluation

What we are not prioritizing right now:
- making BM25 a mainline retriever
- more hybrid tuning before reranking
- LLM-in-the-loop retrieval comparison

## Notes

This repository still keeps BM25 and hybrid because they are useful experimental controls.

But the architectural direction is now clearer:
- dense is the serving retriever
- reranking is the next quality layer
- grounding and citation correctness are the next safety layer
