"""Dual retrieval + Reciprocal Rank Fusion (TRD FR-6/8/9, PHASE_2 B2-B5).

Loads the Phase-A artifacts and builds the candidate pool the scorer ranks:
FAISS (semantic) top-k + BM25 (lexical) top-k, fused by RRF. Imports only
``faiss`` / ``rank_bm25`` / numpy / pandas — never the embedder — so it is safe
on the timed, offline ``rank.py`` path.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aptus.errors import ArtifactError


@dataclass
class Artifacts:
    """All Phase-A artifacts needed by Phase-B ranking."""

    index: faiss.Index
    bm25: object  # rank_bm25.BM25Okapi
    id_map: dict[int, str]  # faiss position -> candidate_id
    features: pd.DataFrame  # indexed by candidate_id
    facts: pd.DataFrame  # indexed by candidate_id
    honeypot_ids: set[str]
    jd_vector: NDArray[np.float32]  # (D,)


def load_artifacts(artifacts_dir: str | Path) -> Artifacts:
    """Load every artifact from ``artifacts_dir`` (fail loud if any is missing)."""
    d = Path(artifacts_dir)
    required = [
        "faiss.index",
        "bm25.pkl",
        "candidate_features.parquet",
        "candidate_facts.parquet",
        "id_map.json",
        "honeypot_ids.json",
        "jd_embedding.npy",
    ]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        raise ArtifactError(f"missing artifacts in {d}: {missing}")

    index = faiss.read_index(str(d / "faiss.index"))
    with (d / "bm25.pkl").open("rb") as fh:
        bundle = pickle.load(fh)
    bm25 = bundle["bm25"]
    id_map = {int(k): v for k, v in json.loads((d / "id_map.json").read_text()).items()}
    features = pd.read_parquet(d / "candidate_features.parquet").set_index("candidate_id")
    facts = pd.read_parquet(d / "candidate_facts.parquet").set_index("candidate_id")
    honeypot_ids = set(json.loads((d / "honeypot_ids.json").read_text()))
    jd_vector = np.load(d / "jd_embedding.npy").astype(np.float32)

    if not (len(id_map) == len(features) == len(facts) == index.ntotal):
        raise ArtifactError(
            f"artifact length mismatch: id_map={len(id_map)} features={len(features)} "
            f"facts={len(facts)} faiss={index.ntotal}"
        )
    return Artifacts(index, bm25, id_map, features, facts, honeypot_ids, jd_vector)


def faiss_topk(index: faiss.Index, jd_vector: NDArray[np.float32], k: int) -> list[int]:
    """Return the top-k FAISS positions for the JD vector (cosine via IndexFlatIP)."""
    q = jd_vector.reshape(1, -1).astype(np.float32)
    _, idx = index.search(q, k)
    return [int(p) for p in idx[0] if p >= 0]


def bm25_topk(bm25: object, query_tokens: list[str], k: int) -> list[int]:
    """Return the top-k BM25 document positions for the tokenized JD query."""
    scores = np.asarray(bm25.get_scores(query_tokens))  # type: ignore[attr-defined]
    if scores.size == 0:
        return []
    top = np.argsort(scores)[::-1][:k]
    return [int(p) for p in top]


def rrf_merge(ranked_lists: list[list[int]], rrf_k: int, pool_size: int) -> list[int]:
    """Reciprocal Rank Fusion over several ranked position lists (FR-9).

    score(p) = Σ_lists 1 / (rrf_k + rank), rank 1-based. Dedup, sort desc, keep
    ``pool_size``. Ties broken by position ascending for determinism.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, pos in enumerate(ranked, start=1):
            scores[pos] = scores.get(pos, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda p: (-scores[p], p))
    return ordered[:pool_size]


def retrieve(
    jd_vector: NDArray[np.float32],
    jd_tokens: list[str],
    index: faiss.Index,
    bm25: object,
    faiss_k: int,
    bm25_k: int,
    rrf_k: int,
    pool_size: int,
) -> list[int]:
    """Return the fused candidate pool (FAISS + BM25 via RRF) as faiss positions."""
    faiss_ids = faiss_topk(index, jd_vector, faiss_k)
    bm25_ids = bm25_topk(bm25, jd_tokens, bm25_k)
    return rrf_merge([faiss_ids, bm25_ids], rrf_k, pool_size)
