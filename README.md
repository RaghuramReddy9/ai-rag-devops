# AI RAG DevOps

A research-oriented RAG benchmark repo for comparing retrieval stacks and measuring whether retrieval gains actually improve final answers.

The project now has two permanent experiment modes:

- Retrieval-only mode
  - compare retrieval stacks without any LLM calls
  - measure `MRR`, `Recall@k`, overlap, and retrieval latency
- Answer-generation mode
  - keep the same dataset, prompt, LLM, and answer schema
  - change only the retriever
  - measure answer quality, grounding, unsupported-risk, and end-to-end latency

## Current Serving Choice

The current best retrieval stack is:

- `dense_rerank`
  - `DenseRetriever`
  - deeper candidate fetch
  - `CrossEncoderReranker`
  - final top-k to generation

BM25 and hybrid were evaluated and kept in the repo as research artifacts, not as serving defaults.

## Final Results

### Retrieval-Only

| Stack | MRR | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| Dense | 0.674 | 0.486 | 0.847 | 0.847 |
| Hybrid | 0.705 | 0.590 | 0.753 | 0.753 |
| Dense + Rerank | 0.817 | 0.694 | 0.809 | 0.917 |

### Answer Generation

| Stack | Correctness | Grounded Citations | Unsupported Risk | Total Avg Latency |
|---|---:|---:|---:|---:|
| Dense + LLM | 0.676 | 1.000 | 0.062 | 31812.55 ms |
| Dense + Rerank + LLM | 0.687 | 1.000 | 0.042 | 17920.62 ms |

Interpretation:
- `dense_rerank` is the best overall stack
- reranking improved retrieval quality and slightly improved final answer quality
- grounded citations remained strong
- retrieval became slower, but total latency was still lower in the completed answer benchmark run

## What We Learned

- Dense retrieval is the strongest base retriever in this corpus
- BM25 adds little useful diversity here and does not justify being part of the serving path
- Hybrid improves early ranking but not enough overall coverage to beat the dense+rerank stack
- Cross-encoder reranking is the highest-value next layer after dense retrieval

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

## Next Improvements

The best next improvements are:

1. stronger grounding and citation enforcement
2. more robust provider retry/resume for long answer runs
3. more careful answer evaluation than the current lightweight scorecard
4. broader datasets beyond the current 4-document benchmark
