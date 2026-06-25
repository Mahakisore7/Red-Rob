# Eval Report (PHASE_4)

_Generated 2026-06-26. Gold set: 180 **weak/heuristic** labels (S1-correlated; hand-correct before trusting). LLM run over top-30 of composite._

| ranker | NDCG@10 | NDCG@50 | MAP | P@10 | composite | honeypot |
|---|---|---|---|---|---|---|
| naive (AI-count x response) | 0.442 | 0.551 | 0.635 | 0.400 | 0.502 | 0.17 |
| title-only (tier) | 0.927 | 0.885 | 0.773 | 0.900 | 0.890 | 0.17 |
| composite (S1-S5, w=1.0) | 1.000 | 0.890 | 0.849 | 1.000 | 0.944 | 0.17 |
| composite+LLM (w=0.70) | 1.000 | 0.894 | 0.856 | 1.000 | 0.947 | 0.17 |
| composite+LLM (w=0.40) | 1.000 | 0.895 | 0.859 | 1.000 | 0.947 | 0.17 |

## Decision Rule 1
Best NDCG@10 at **w=1.00** (1.000). Ship the smallest (1-w) that maximises holdout NDCG@10; tie -> prefer composite (w=1.0).
