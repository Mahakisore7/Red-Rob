"""Composite scoring of the retrieved pool (PHASE_2 B6/B9).

Turns the RRF pool into a ranked top-N. For each pooled candidate it computes S1
(JD cosine via the reconstructed FAISS vector), looks up the precomputed S2-S5 +
modifiers + flags, applies the master formula, and sorts deterministically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from aptus import signals
from aptus.retriever import Artifacts


@dataclass
class ScoredCandidate:
    """One ranked candidate."""

    candidate_id: str
    score: float
    s1: float


def score_pool(art: Artifacts, positions: list[int], top_n: int) -> list[ScoredCandidate]:
    """Score every pooled position and return the deterministic top-N.

    Sort key is ``(-score, candidate_id)`` so ties break by candidate_id ascending
    (matches the validator's tie-break rule).
    """
    jd = art.jd_vector.astype(np.float32)
    scored: list[ScoredCandidate] = []
    for pos in positions:
        cid = art.id_map[pos]
        if cid not in art.features.index:
            continue
        vec = art.index.reconstruct(pos)
        cosine = float(np.dot(jd, vec))
        s1 = signals.s1_semantic(cosine)
        feat = cast("Mapping[str, object]", art.features.loc[cid])
        score = signals.final_score(s1, feat)
        scored.append(ScoredCandidate(cid, score, s1))

    scored.sort(key=lambda r: (-r.score, r.candidate_id))
    return scored[:top_n]
