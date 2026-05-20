# AI Retrieval Inspector

An AI engineering dashboard and benchmark system for inspecting retrieval quality, evidence grounding, and latency tradeoffs in RAG pipelines. This is not a chatbot project. It is an evidence-driven retrieval observability project focused on understanding why RAG systems succeed or fail.

The project now has two permanent experiment modes:

- Retrieval-only mode
  - compare retrieval stacks without any LLM calls
  - measure `MRR`, `Recall@k`, overlap, and retrieval latency
- Answer-generation mode
  - keep the same dataset, prompt, LLM, and answer schema
  - change only the retriever
  - measure answer quality, grounding, unsupported-risk, and end-to-end latency

## Showcase: AI Retrieval Inspector

`AI Retrieval Inspector` is a dashboard layer on top of the benchmark system. It is designed to help developers understand why a RAG system succeeds or fails by making retrieval behavior, evidence quality, citations, latency, and grounding risk visible in one place.

This is intentionally not framed as a chatbot. The core product idea is retrieval observability: before trusting an answer, inspect the evidence path that produced it.

### Project Problem

RAG systems often fail in ways that are hard to debug from the final answer alone. A response can look plausible while being weakly grounded, missing the expected source chunk, over-relying on irrelevant context, or hiding latency tradeoffs behind a single generation call.

This project addresses that problem by separating the benchmark backend from a lightweight inspection UI:

- benchmark retrievers under controlled conditions
- compare dense, BM25, hybrid, and dense-rerank retrieval outputs
- inspect the exact chunks surfaced for a query
- evaluate whether citations map back to retrieved evidence
- compare quality and latency tradeoffs using saved experiment outputs

### Why Retrieval Observability Matters

For production AI engineering, retrieval quality is not just an offline metric. It affects answer correctness, hallucination risk, latency, cost, and developer trust.

The inspector makes the retrieval layer explainable by showing:

- which retriever found which source chunks
- whether reranking changed the evidence order
- whether the answer cites retrieved chunks
- whether unsupported-risk is rising or falling across benchmark runs
- where latency is spent across retrieval and generation

This gives teams a practical way to reason about failure modes instead of treating the LLM response as a black box.

### Showcase Architecture

```text
Existing benchmark backend
        |
        v
Read-only experiment artifacts
        |
        v
src/showcase demo wrapper
        |
        v
Streamlit dashboard
```

The showcase layer reads from existing configs and `experiments/results/` artifacts where possible. It can also attempt live retriever calls through the existing retrieval factory, but falls back to saved benchmark outputs when model loading, API keys, or local runtime dependencies are unavailable.

No retrieval or evaluation modules are refactored for the UI. The dashboard is a thin presentation and inspection layer over the benchmark system.

### Dashboard Sections

- System Overview: run mode, grounding status, latency, and architecture flow
- Query Input: selectable benchmark questions or custom inspection query
- Retriever Comparison: dense, BM25, hybrid, and dense-rerank side-by-side
- Retrieved Evidence Explorer: expandable source/chunk cards for ranked evidence
- Answer + Citations: final answer artifact from answer-generation results when available
- Grounding / Risk Analysis: citation-to-context checks and hallucination-risk signals
- Benchmark Insights: retrieval metrics, answer quality, unsupported-risk, and latency summaries

### Local Setup

Use the local project environment:

```bash
uv sync
```

Expected environment variables for live answer generation:

```bash
OPENROUTER_API_KEY=
OPENROUTER_MODEL=
```

The dashboard can still run without an API key. In that case it uses saved benchmark outputs from `experiments/results/` and displays unavailable live metrics as `not available`.

### Demo Instructions

Start the showcase dashboard:

```bash
uv run streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501/
```

Recommended showcase flow:

- start with the default saved benchmark question
- compare dense, BM25, hybrid, and dense-rerank retrieval results
- open the evidence cards and inspect source/chunk IDs
- review the answer citation trail
- check the grounding/risk panel
- finish with benchmark insights for quality and latency tradeoffs

### Streamlit Community Cloud Deployment

Use these settings when creating the app in Streamlit Community Cloud:

```text
Repository: RaghuramReddy9/ai-rag-devops
Branch: main
Main file path: streamlit_app.py
Python version: 3.11+
```

The root `requirements.txt` is intentionally lightweight for the public dashboard:

- `streamlit`
- `pyyaml`

The deployed showcase uses saved benchmark artifacts from `experiments/results/` by default, so it does not need API keys or local model downloads to render. Leave the live retriever and live LLM toggles off in the public demo unless the deployment environment is provisioned with the full benchmark dependencies and required secrets.

If live answer generation is enabled, add secrets in Streamlit Cloud rather than committing them:

```toml
OPENROUTER_API_KEY = "..."
OPENROUTER_MODEL = "..."
```

### Screenshot Placeholders

Add screenshots after running the Streamlit app locally:

```text
docs/screenshots/retrieval-inspector-overview.png
docs/screenshots/retriever-comparison.png
docs/screenshots/evidence-explorer.png
docs/screenshots/grounding-risk-panel.png
```

Suggested captions:

- System overview with architecture flow and run status
- Retriever comparison table across dense, BM25, hybrid, and dense-rerank
- Expandable evidence cards showing source and chunk IDs
- Grounding and hallucination-risk panel with citation checks

### Benchmark Findings

The completed benchmark supports `dense_rerank` as the preferred serving pipeline for this corpus:

- dense retrieval was the strongest base retriever
- BM25 contributed limited useful diversity
- hybrid improved early ranking but did not beat dense-rerank overall
- cross-encoder reranking improved top-result quality and `Recall@5`
- answer-generation experiments showed lower unsupported-risk for dense-rerank

The key production conclusion is that retrieval-only gains should be validated against answer quality and latency. In this project, reranking added retrieval-side cost but improved the final answer profile enough to justify keeping it as the preferred stack.

### Production Engineering Considerations

- Evidence visibility: every answer should be inspectable through source chunks and citation metadata.
- Repeatable evaluation: saved JSONL and summary artifacts make comparisons reproducible.
- Failure isolation: retrieval-only runs separate retriever failures from LLM-generation failures.
- Latency accounting: retrieval, generation, and total latency are tracked separately.
- Fallback behavior: the dashboard remains useful from saved artifacts even when live API calls are unavailable.
- Secret handling: API keys are read from environment variables and are not surfaced in the UI.
- Serving discipline: BM25 and hybrid remain benchmark artifacts unless evidence supports promoting them.

### Future Roadmap

- Add per-query diff views showing how reranking changes dense retrieval order.
- Add source coverage and duplicate-evidence diagnostics.
- Add calibrated confidence only if derived from validated evaluation data.
- Add regression checks for retrieval quality across corpus or embedding changes.
- Add exportable inspection reports for individual benchmark questions.
- Add support for comparing additional rerankers or embedding models under the same evaluation harness.

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
