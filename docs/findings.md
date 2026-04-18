# Findings

## Why This Project Matters

This repository is not just a RAG demo.

It shows a practical retrieval research workflow:
- build a shared corpus
- benchmark multiple retrievers on the same gold set
- measure where each retriever helps or hurts
- use those findings to guide architecture decisions

That makes the project valuable because it demonstrates:
- experimental discipline
- retrieval evaluation, not just app wiring
- evidence-based design decisions
- a clear roadmap from retrieval to reranking to grounded generation

## Corpus And Benchmark

Current setup:
- `4` source documents in `data/raw`
- `69` chunks in [chunks.jsonl](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/data/processed/chunks.jsonl)
- `48` manually labeled gold questions in [gold_qa.jsonl](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/data/eval/gold_qa.jsonl)

Question mix:
- direct questions
- keyword-heavy questions
- paraphrased / semantic questions
- multi-evidence questions
- distractor questions

This benchmark shape makes the repo stronger than a simple "ask one README question" demo.

## Retrieval Results

Current retrieval-only benchmark:

### Dense

- `MRR: 0.674`
- `Recall@1: 0.486`
- `Recall@3: 0.847`
- `Recall@5: 0.847`

Source:
- [dense_retrieval_summary.json](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/experiments/results/dense_retrieval_summary.json)

### BM25

- `MRR: 0.496`
- `Recall@1: 0.392`
- `Recall@3: 0.576`
- `Recall@5: 0.576`

Source:
- [bm25_retrieval_summary.json](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/experiments/results/bm25_retrieval_summary.json)

### Hybrid

- `MRR: 0.705`
- `Recall@1: 0.59`
- `Recall@3: 0.753`
- `Recall@5: 0.753`

Source:
- [hybrid_retrieval_summary.json](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/experiments/results/hybrid_retrieval_summary.json)

## What We Observed

### Observation 1

Dense is the strongest retriever for coverage.

Evidence:
- best `Recall@3`
- best `Recall@5`

Interpretation:
- dense is retrieving relevant chunks more consistently within the top few results
- this makes it the right primary retriever for the main pipeline

### Observation 2

Hybrid improves early ranking, but not overall coverage.

Evidence:
- best `MRR`
- best `Recall@1`
- worse than dense at `Recall@3` and `Recall@5`

Interpretation:
- hybrid helps get a relevant chunk earlier
- but it does not beat dense on broader top-k coverage
- this makes hybrid useful as an experiment artifact, not as the main serving retriever

### Observation 3

BM25 is weak on this corpus.

Evidence:
- lowest `MRR`
- lowest `Recall@1`
- lowest `Recall@3`
- lowest `Recall@5`

Interpretation:
- BM25 is still useful as a baseline and diagnostic
- but the benchmark does not justify making it part of the main retrieval pipeline

## Overlap Findings

Dense vs BM25 overlap analysis at top-5:

- `avg_overlap@5 = 2.104`
- `BM25 adds a new relevant chunk on 6.2% of queries`
- `BM25 is useless on 93.8% of queries`

Source:
- [retrieval_overlap_summary.json](/C:/RRR/projects/ai-engineer-portfolio/ai-rag-devops/experiments/results/retrieval_overlap_summary.json)

Interpretation:
- BM25 is not contributing much useful retrieval diversity
- hybrid underperformance is not just a random metric artifact
- dense should remain the primary retriever

## Category-Level Finding

One important result:

- on `keyword_heavy` questions, BM25 adds a new relevant chunk on `0.0%` of queries

That matters because BM25 would normally be expected to help most on lexical questions.

Interpretation:
- the current corpus and benchmark do not support BM25 as a high-value retrieval branch
- the project's next gains are more likely to come from reranking than from more BM25 tuning

## Architecture Decision

Based on the benchmark evidence:

- dense stays the main application retriever
- BM25 stays as a baseline and diagnostic tool
- hybrid stays as a research artifact

This is a strong project signal because the architecture is being shaped by measured evidence, not by assumptions.

## What Comes Next

The next meaningful steps are:

1. Add cross-encoder reranking on top of dense retrieval
2. Add grounding and citation enforcement
3. Then evaluate end-to-end answer quality

This progression is important because:
- retrieval quality should be stable first
- ranking quality should improve second
- grounded answer behavior should be enforced third
- only then should answer quality be judged seriously

## Portfolio Value

If you show this project, the important story is:

- you did not stop at "RAG works"
- you built a benchmark
- you compared retrievers fairly
- you measured overlap and retrieval diversity
- you used the evidence to simplify the serving architecture
- you identified reranking and grounding as the next high-value layers

That is a much stronger story than a basic chatbot repo.
