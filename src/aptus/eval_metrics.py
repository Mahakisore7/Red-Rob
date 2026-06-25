"""Ranking metrics for the eval harness (docs/05 §2, PHASE_4).

Hand-rolled and fixture-tested so the numbers are transparent and defensible:
NDCG@k (linear gain), MAP, P@k, and the organizer's composite challenge score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Relevance >= this counts as "relevant" for MAP / P@k (docs/05 §2).
RELEVANT_THRESHOLD = 2


def dcg(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain over the first k items (linear gain)."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked: Sequence[float], ideal: Sequence[float], k: int) -> float:
    """NDCG@k: DCG of the ranked relevances over the ideal (best-possible) DCG."""
    ideal_sorted = sorted(ideal, reverse=True)
    idcg = dcg(ideal_sorted, k)
    if idcg == 0:
        return 0.0
    return dcg(ranked, k) / idcg


def precision_at_k(ranked: Sequence[float], k: int, threshold: int = RELEVANT_THRESHOLD) -> float:
    """Fraction of the top-k that are relevant (relevance >= threshold)."""
    if k <= 0:
        return 0.0
    hits = sum(1 for rel in ranked[:k] if rel >= threshold)
    return hits / k


def average_precision(
    ranked: Sequence[float],
    total_relevant: int,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """Average precision: mean of precision@k at each relevant hit.

    ``total_relevant`` is the number of relevant items in the whole gold set (the
    AP denominator), so missing a relevant item is penalized.
    """
    if total_relevant <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, rel in enumerate(ranked, start=1):
        if rel >= threshold:
            hits += 1
            score += hits / i
    return score / total_relevant


def composite_challenge(ndcg10: float, ndcg50: float, map_: float, p10: float) -> float:
    """Organizer's weighted score: 0.50 NDCG@10 + 0.30 NDCG@50 + 0.15 MAP + 0.05 P@10."""
    return 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_ + 0.05 * p10
