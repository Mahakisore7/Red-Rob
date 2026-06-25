"""Determinism + LLM-blend pipeline tests (NFR-5, PHASE_3).

Builds artifacts with a fake embedder, then runs the ranking pipeline with a fake
LLM generator: asserts the LLM path produces grounded reasoning + a valid CSV, and
that two runs are byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from aptus.cli.precompute import run_precompute
from aptus.cli.rank import run_rank

_DIM = 32


class FakeEmbedder:
    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / (float(np.linalg.norm(v)) or 1.0)
        return out


def _fake_llm(_: str) -> str:
    # grounded (anchors to the real title) and parseable
    return '{"fit_score": 82, "hire_recommendation": "yes", "reasoning": "Senior ML Engineer fit."}'


def _build_artifacts(tmp_path: Path, record_factory: Callable[..., dict]) -> Path:  # type: ignore[type-arg]
    recs = []
    for i in range(1, 121):
        r = record_factory(f"CAND_{i:07d}")
        r["profile"]["summary"] += f" marker {i}"
        recs.append(r)
    art_dir = tmp_path / "artifacts"
    run_precompute(iter(recs), art_dir, FakeEmbedder())
    return art_dir


def test_llm_pipeline_produces_grounded_rows(
    tmp_path: Path,
    record_factory: Callable[..., dict],  # type: ignore[type-arg]
) -> None:
    art_dir = _build_artifacts(tmp_path, record_factory)
    summary = run_rank(
        art_dir, tmp_path / "sub.csv", validate=True, use_llm=True, llm_generate=_fake_llm
    )
    assert summary["rows"] == 100
    assert summary["validator"] == "passed"
    assert int(summary["llm_reasoned"]) > 0  # grounded LLM reasoning was used
    assert summary["llm_weight"] == 0.70


def test_rank_byte_identical_across_runs(
    tmp_path: Path,
    record_factory: Callable[..., dict],  # type: ignore[type-arg]
) -> None:
    art_dir = _build_artifacts(tmp_path, record_factory)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    run_rank(art_dir, a, validate=False, use_llm=True, llm_generate=_fake_llm)
    run_rank(art_dir, b, validate=False, use_llm=True, llm_generate=_fake_llm)
    assert a.read_bytes() == b.read_bytes()
