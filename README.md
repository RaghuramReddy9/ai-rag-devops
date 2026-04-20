# AI RAG DevOps

A production-grade RAG benchmarking system for comparing retrieval pipelines and testing whether retrieval gains hold up in end-to-end answer generation. This is not a chatbot project. It is an evidence-driven benchmark repo for retrieval quality, grounding, and latency tradeoffs.

The project now has two permanent experiment modes:

- Retrieval-only mode
  - compare retrieval stacks without any LLM calls
  - measure `MRR`, `Recall@k`, overlap, and retrieval latency
- Answer-generation mode
  - keep the same dataset, prompt, LLM, and answer schema
  - change only the retriever
  - measure answer quality, grounding, unsupported-risk, and end-to-end latency

## Final Decision

The chosen production retrieval pipeline is:

- `dense_rerank`
  - `DenseRetriever`
  - deeper candidate fetch
  - `CrossEncoderReranker`
  - final top-k to generation

Why this is the default:
- dense outperformed BM25 on this corpus
- hybrid RRF did not improve recall enough because BM25 added little useful diversity
- dense + cross-encoder reranking delivered the strongest overall retrieval results

BM25 and hybrid remain in the repo as research artifacts, not serving defaults.

## Final Findings

### Retrieval-Only Benchmark

| Stack | MRR | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| Dense | 0.674 | 0.486 | 0.847 | 0.847 |
| BM25 | 0.496 | 0.392 | 0.576 | 0.576 |
| Hybrid | 0.705 | 0.590 | 0.753 | 0.753 |
| Dense + Rerank | 0.817 | 0.694 | 0.809 | 0.917 |

Interpretation:
- dense is the strongest base retriever
- reranking significantly improved early-rank relevance
- reranking improved top-result quality substantially
- reranking improved `Recall@5`
- `Recall@3` dipped slightly, but dense+rerank is still the best retrieval design overall

### Answer-Generation Benchmark

| Stack | Correctness | Grounded Citations | Unsupported Risk | Retrieval Avg | Generation Avg | Total Avg |
|---|---:|---:|---:|---:|---:|---:|
| Dense + LLM | 0.676 | 1.000 | 0.062 | 156.63 ms | 31655.85 ms | 31812.55 ms |
| Dense + Rerank + LLM | 0.687 | 1.000 | 0.042 | 458.31 ms | 17462.25 ms | 17920.62 ms |

Interpretation:
- answer-generation experiments keep the same dataset, prompt, model, and answer schema and change only the retriever
- the current answer scorecard is lightweight but complete enough to compare `dense + LLM` against `dense_rerank + LLM`
- `dense_rerank` slightly improved correctness, kept citation grounding perfect, reduced unsupported-risk, and finished with lower total latency in the completed run

## What We Learned

- Dense retrieval is the strongest base retriever in this corpus
- BM25 adds little useful diversity here and does not justify being part of the serving path
- Hybrid improves early ranking but not enough overall coverage to beat the dense+rerank stack
- Cross-encoder reranking is now part of the preferred serving stack

### Why Hybrid Failed

- BM25 overlapped heavily with dense retrieval and contributed little new relevant evidence
- overlap analysis showed BM25 added a new relevant chunk on only `6.2%` of queries
- that low contribution meant hybrid RRF did not improve recall enough to justify the added complexity

### Failure Patterns That Still Matter

- dense fails, rerank fixes:
  cross-encoder reranking often recovers the right chunk when dense candidates are present but badly ordered
- both fail:
  multi-evidence questions are still the hardest cases
- lexical retrieval contributes little:
  BM25 remains useful as a baseline and diagnostic, but not as part of the serving path

Concrete examples:
- `gold_002`: dense missed the right LangChain definition chunk; reranking promoted it and fixed the answer
- `gold_037`: both systems remained weak on a multi-evidence Transformers question
- `gold_029`: reranking did not help enough to change the outcome

## Repository Layout

```text
.
|-- configs/
|   |-- default.yaml
|   |-- dense.yaml
|   |-- dense_answers.yaml
|   |-- dense_answers_sample.yaml
|   |-- dense_rerank.yaml
|   |-- dense_rerank_answers.yaml
|   |-- dense_rerank_answers_sample.yaml
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
|   |-- architecture.md
|   `-- findings.md
|-- experiments/
|   `-- results/
|-- scripts/
|   `-- smoke_pipeline.py
|-- src/
|   |-- common/
|   |-- embeddings/
|   |-- eval/
|   |-- generation/
|   |-- ingestion/
|   |-- retrieval/
|   `-- pipeline.py
`-- pyproject.toml
```

## Experiment Modes

### 1. Retrieval-Only

Use this for:
- `dense`
- `hybrid`
- `dense_rerank`
- optional BM25 diagnostics

Per question:
- retrieve chunks
- save citations and retrieved context
- no LLM call

Useful commands:

```bash
uv run python -m src.eval.run_predictions --config configs/dense.yaml
uv run python -m src.eval.retrieval_eval --config configs/dense.yaml

uv run python -m src.eval.run_hybrid_predictions --config configs/hybrid.yaml
uv run python -m src.eval.retrieval_eval --config configs/hybrid.yaml

uv run python -m src.eval.run_dense_rerank_predictions --config configs/dense_rerank.yaml
uv run python -m src.eval.retrieval_eval --config configs/dense_rerank.yaml

uv run python -m src.eval.analyze_retrieval_overlap --k 5
```

### 2. Answer Generation

Use this for:
- `dense + LLM`
- `dense_rerank + LLM`

Per question:
- retrieve chunks
- build prompt
- call the LLM once
- save answer, citations, retrieved context, and latency

This mode exists to judge whether reranking's answer-quality gain is worth its latency cost.

Small sanity runs:

```bash
uv run python -m src.eval.run_dense_answers --config configs/dense_answers_sample.yaml
uv run python -m src.eval.run_dense_rerank_answers --config configs/dense_rerank_answers_sample.yaml
```

Full runs:

```bash
uv run python -m src.eval.run_dense_answers --config configs/dense_answers.yaml
uv run python -m src.eval.run_dense_rerank_answers --config configs/dense_rerank_answers.yaml
```

Compare latency:

```bash
uv run python -m src.eval.summarize_answer_latency --inputs experiments/results/dense_answers.jsonl experiments/results/dense_rerank_answers.jsonl --output experiments/results/final_answer_latency_summary.json
```

Compare answer quality:

```bash
uv run python -m src.eval.evaluate_answer_scorecard --inputs experiments/results/dense_answers.jsonl experiments/results/dense_rerank_answers.jsonl --output experiments/results/final_answer_scorecard_summary.json
```

## Latency Tradeoff

This project tracks:
- retrieval latency
- rerank latency as part of retrieval-side cost
- generation latency
- total latency

The point is not just to improve retrieval metrics. The real question is whether reranking improves final answers enough to justify its latency cost.

## Why BM25 And Hybrid Stay In The Repo

They still matter as benchmark artifacts:

- BM25
  - lexical baseline
  - overlap and diversity diagnostics
- Hybrid
  - useful to test whether fusing lexical and dense retrieval helps first-hit ranking

But they are not serving defaults because the benchmark evidence does not support that choice.

## Key Artifacts

- Retrieval overlap:
  - `experiments/results/retrieval_overlap_summary.json`
- Final answer scorecard:
  - `experiments/results/final_answer_scorecard_summary.json`
- Final answer latency:
  - `experiments/results/final_answer_latency_summary.json`
- Architecture notes:
  - `docs/architecture.md`
- Findings summary:
  - `docs/findings.md`

## Environment

Use the local project environment only:

```bash
uv run ...
```

Expected environment variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` optional override

Dense embeddings are local and do not require a hosted embedding API.

## Final Project Position

This project closes with a clear result:

- `dense_rerank` is the preferred serving stack for this corpus
- `bm25` and `hybrid` remain in the repo as benchmark artifacts
- retrieval-only and answer-generation benchmarking are both implemented and completed

The value of the project is not just that it built a RAG pipeline. It showed, with controlled experiments, which retrieval design held up best once answer quality and latency were measured together.
