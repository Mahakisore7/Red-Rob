"""End-to-end precompute test with a fake embedder (no model needed).

Doubles as the Phase-1 subset validation: runs the full A1-A10 pipeline on a few
records and asserts every artifact exists with the right shape + the invariants.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from aptus.cli.precompute import build_jd_query, run_precompute

_DIM = 32


class FakeEmbedder:
    """Deterministic, normalized stand-in for bge-large (per-text seeded)."""

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            nrm = float(np.linalg.norm(v)) or 1.0
            out[i] = v / nrm
        return out


def test_run_precompute_end_to_end(
    tmp_path: Path,
    record_factory: Callable[..., dict],  # type: ignore[type-arg]
) -> None:
    recs = [record_factory(f"CAND_000000{i}") for i in range(1, 5)]
    # Make the 4th a honeypot: two expert skills with zero duration (FR-7a).
    recs[3]["skills"] = [
        {"name": "A", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "B", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ]

    summary = run_precompute(iter(recs), tmp_path, FakeEmbedder())
    assert summary["n"] == 4
    assert summary["dim"] == _DIM

    for name in (
        "faiss.index",
        "bm25.pkl",
        "candidate_features.parquet",
        "candidate_facts.parquet",
        "id_map.json",
        "honeypot_ids.json",
        "jd_embedding.npy",
    ):
        assert (tmp_path / name).exists(), f"missing artifact {name}"

    idx = faiss.read_index(str(tmp_path / "faiss.index"))
    assert idx.ntotal == 4

    feats = pd.read_parquet(tmp_path / "candidate_features.parquet")
    assert len(feats) == 4
    assert {"candidate_id", "s2_career_arc", "is_honeypot"}.issubset(feats.columns)

    facts_df = pd.read_parquet(tmp_path / "candidate_facts.parquet")
    assert len(facts_df) == 4

    id_map = json.loads((tmp_path / "id_map.json").read_text(encoding="utf-8"))
    assert len(id_map) == 4

    hps = json.loads((tmp_path / "honeypot_ids.json").read_text(encoding="utf-8"))
    assert "CAND_0000004" in hps

    jd = np.load(tmp_path / "jd_embedding.npy")
    assert jd.shape == (_DIM,)

    with (tmp_path / "bm25.pkl").open("rb") as fh:
        bundle = pickle.load(fh)
    assert "bm25" in bundle and "id_map" in bundle


def test_build_jd_query_mentions_core_concepts() -> None:
    q = build_jd_query().lower()
    assert "retrieval" in q and "ranking" in q and "python" in q
