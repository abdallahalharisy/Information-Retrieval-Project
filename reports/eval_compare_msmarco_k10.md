# IR Evaluation Report

- Dataset: `msmarco`
- Dataset page: https://ir-datasets.com/msmarco-document.html#msmarco-document/dev
- Qrels file: https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz
- Queries file: https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz
- Qrels query count: `5193`
- Queries used for evaluation: `3`
- Query limit: `3`
- K: `10`
- Created at: `2026-06-21T01:38:34`

## Summary Table

| Method | Mode | Qrels Queries | Used Queries | Evaluated Queries | MAP | Precision@10 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | baseline | 5193 | 3 | 3 | 0.6667 | 0.0667 | 0.6667 | 0.6667 |
| bm25 | enhanced | 5193 | 3 | 3 | 0.6667 | 0.0667 | 0.6667 | 0.6667 |
| tfidf | baseline | 5193 | 3 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tfidf | enhanced | 5193 | 3 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rrf | baseline | 5193 | 3 | 3 | 0.0370 | 0.0333 | 0.3333 | 0.1003 |
| rrf | enhanced | 5193 | 3 | 3 | 0.0370 | 0.0333 | 0.3333 | 0.1003 |
| bert | baseline | 5193 | 3 | 3 | 0.1111 | 0.0333 | 0.3333 | 0.1667 |
| bert | enhanced | 5193 | 3 | 3 | 0.1111 | 0.0333 | 0.3333 | 0.1667 |

## Enhanced vs Baseline Deltas

| Method | Delta MAP | Delta Precision@10 | Delta Recall@10 | Delta nDCG@10 |
|---|---:|---:|---:|---:|
| bm25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tfidf | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rrf | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bert | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation Guide

- `baseline`: query normalization without extra synonym expansion, spell correction, or history boosting.
- `enhanced`: query refinement enabled using synonyms, spell correction, and configured history boost.
- Positive deltas indicate that the additional features improved the metric.
- All qrels query IDs are used by default. Very low scores can happen if qrels reference relevant documents outside the currently processed local corpus subset.

## Notes

- The project currently uses MSMARCO only; FEVER was removed per the updated client requirement.
- BERT/Word2Vec are included only when `include_embeddings=true` or `--include-embeddings` is used.
