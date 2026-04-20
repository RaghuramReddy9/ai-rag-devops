# Findings

## Benchmark Scope

Current benchmark:
- `4` source documents
- `69` chunks
- `48` manually labeled questions

Question types:
- direct
- keyword-heavy
- paraphrased / semantic
- multi-evidence
- distractor

This repo is valuable because it does not stop at "RAG works." It compares retrieval stacks fairly and then checks whether retrieval improvements actually improve final answers.

## Retrieval Findings

### Dense

- `MRR: 0.674`
- `Recall@1: 0.486`
- `Recall@3: 0.847`
- `Recall@5: 0.847`

### Hybrid

- `MRR: 0.705`
- `Recall@1: 0.590`
- `Recall@3: 0.753`
- `Recall@5: 0.753`

### Dense + Rerank

- `MRR: 0.817`
- `Recall@1: 0.694`
- `Recall@3: 0.809`
- `Recall@5: 0.917`

## Retrieval Conclusion

Dense+rerank is the strongest retrieval stack.

Why:
- best `MRR`
- best `Recall@1`
- best `Recall@5`

Interpretation:
- dense gives the strongest candidate pool
- the cross-encoder improves final ordering enough to outperform the other stacks

## Why BM25 Failed As A Serving Path

Overlap findings:
- `avg_overlap@5 = 2.104`
- BM25 adds a new relevant chunk on `6.2%` of queries
- BM25 is useless on `93.8%` of queries

Important category result:
- on `keyword-heavy` questions, BM25 adds a new relevant chunk on `0.0%` of queries

Conclusion:
- BM25 stays useful as a baseline and diagnostic
- BM25 is not useful enough to remain in the serving path

## Why Hybrid Failed As A Serving Path

Hybrid helped:
- early ranking
- `MRR`
- `Recall@1`

Hybrid did not help enough:
- dense had stronger broader coverage than hybrid
- dense+rerank outperformed both

Conclusion:
- hybrid stays in the repo as a research artifact
- it is not the default serving stack

## Answer Benchmark Findings

### Dense + LLM

- `avg_correctness_recall: 0.676`
- `expected_chunk_retrieved_rate: 0.938`
- `citations_grounded_rate: 1.000`
- `unsupported_risk_rate: 0.062`
- `total_avg latency: 31812.55 ms`

### Dense + Rerank + LLM

- `avg_correctness_recall: 0.687`
- `expected_chunk_retrieved_rate: 0.938`
- `citations_grounded_rate: 1.000`
- `unsupported_risk_rate: 0.042`
- `total_avg latency: 17920.62 ms`

## Answer Benchmark Conclusion

Dense+rerank also wins in answer mode.

Why:
- correctness improved slightly
- expected chunk retrieval stayed the same
- citation grounding stayed perfect
- unsupported-risk decreased

## Latency Tradeoff

Dense+rerank adds retrieval-side cost:
- dense retrieval avg: `156.63 ms`
- dense+rerank retrieval avg: `458.31 ms`

But in the completed run:
- dense generation avg: `31655.85 ms`
- dense+rerank generation avg: `17462.25 ms`
- dense total avg: `31812.55 ms`
- dense+rerank total avg: `17920.62 ms`

Interpretation:
- reranking is slower on retrieval
- but the overall end-to-end run was still faster in the completed benchmark
- the main driver was lower generation time, likely because reranked context was cleaner and easier for the model to answer from

## Failure Analysis

### Dense failed, rerank succeeded

Examples:
- `gold_002`
  - dense surfaced weak README chunks and effectively failed
  - dense+rerank promoted the correct LangChain definition chunk
- `gold_017`
  - reranking surfaced the correct LangSmith service-key chunk
- `gold_022`
  - reranking improved answer relevance for the Transformers architecture-adoption question

### Both were weak

Example:
- `gold_037`
  - multi-evidence answer about Transformers design principles remained hard

### Reranking hurt or did not matter

Examples:
- `gold_029`
- `gold_032`
- `gold_044`

Interpretation:
- reranking is not universally better on every single question
- but it wins strongly enough overall to justify the stack choice

## Final Decision

Current default recommendation:

- serving stack: `dense_rerank`
- research artifacts kept in repo:
  - `bm25`
  - `hybrid`

## Portfolio Value

This project is worth showing because it demonstrates:
- benchmark design
- retriever comparison
- reranking integration
- answer-quality evaluation
- latency tradeoff analysis
- evidence-based architecture decisions
