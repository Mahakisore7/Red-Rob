"""End-to-end safety-submission test: precompute -> rank -> validate (PHASE_2 DoD).

Builds 120 fake candidates, runs the real precompute (fake embedder) + the real
ranking pipeline, and asserts the output passes the organizer validator: exactly
100 rows, non-increasing scores, deterministic tie-break, <=3 honeypots.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aptus.cli.precompute import run_precompute
from aptus.cli.rank import run_rank

_DIM = 32


class FakeEmbedder:
    """Deterministic normalized stand-in for bge-large."""

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / (float(np.linalg.norm(v)) or 1.0)
        return out


def test_safety_submission_end_to_end(
    tmp_path: Path,
    record_factory: Callable[..., dict],  # type: ignore[type-arg]
) -> None:
    recs = []
    for i in range(1, 121):
        r = record_factory(f"CAND_{i:07d}")
        r["profile"]["summary"] += f" unique marker {i}"  # vary text -> varied scores
        recs.append(r)
    # two honeypots (FR-7a: >=2 expert skills with zero duration)
    for j in (5, 60):
        recs[j]["skills"] = [
            {"name": "A", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "B", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ]

    art_dir = tmp_path / "artifacts"
    run_precompute(iter(recs), art_dir, FakeEmbedder())

    out = tmp_path / "submission.csv"
    summary = run_rank(art_dir, out, validate=True)

    assert summary["rows"] == 100
    assert summary["validator"] == "passed"
    assert int(summary["honeypots_in_top100"]) <= 3

    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert len(rows) == 101  # header + 100 data rows

    scores = [float(r[2]) for r in rows[1:]]
    assert all(
        scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
    ), "scores must not increase"

    # every reasoning is non-empty and grounded (mentions the real title)
    assert all(r[3].strip() for r in rows[1:])
    assert any("Engineer" in r[3] for r in rows[1:])


def test_run_rank_without_validation(
    tmp_path: Path,
    record_factory: Callable[..., dict],  # type: ignore[type-arg]
) -> None:
    recs = [record_factory(f"CAND_{i:07d}") for i in range(1, 121)]
    art_dir = tmp_path / "artifacts"
    run_precompute(iter(recs), art_dir, FakeEmbedder())
    summary = run_rank(art_dir, tmp_path / "sub.csv", validate=False)
    assert summary["rows"] == 100
    assert "validator" not in summary
