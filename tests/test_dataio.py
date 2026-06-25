"""Streaming loader tests (plain + gzip, missing file)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from aptus import dataio


def _record(cid: str) -> dict:  # type: ignore[type-arg]
    return {"candidate_id": cid, "profile": {"current_title": "ML Engineer"}}


def test_iter_records_plain(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps(_record("CAND_0000001")) + "\n\n", encoding="utf-8"
    )  # blank line skipped
    recs = list(dataio.iter_records(p))
    assert len(recs) == 1
    assert recs[0]["candidate_id"] == "CAND_0000001"


def test_iter_records_gzip_and_count(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl.gz"
    with gzip.open(p, "wb") as fh:
        for i in (1, 2, 3):
            fh.write((json.dumps(_record(f"CAND_000000{i}")) + "\n").encode())
    assert dataio.count_records(p) == 3
    cands = list(dataio.iter_candidates(p))
    assert [c.candidate_id for c in cands] == ["CAND_0000001", "CAND_0000002", "CAND_0000003"]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(dataio.iter_records(tmp_path / "nope.jsonl"))
