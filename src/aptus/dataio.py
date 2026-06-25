"""Streaming JSONL candidate loader (TRD FR-1 / NFR-2).

Lives in the package so the ``aptus-precompute`` console script works when the
project is installed (CI / judge sandbox), not just when the repo root is on the
path. Stream-parses ``candidates.jsonl[.gz]`` one record at a time. The analysis
scripts re-export these helpers from ``scripts.dataio`` for convenience.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, cast

import orjson

from aptus.schema import Candidate

# The organizer ships a plain .jsonl; the loader also transparently handles .gz.
DEFAULT_DATA_PATH = Path("data/candidates.jsonl")


def _open(path: Path) -> IO[bytes]:
    """Open a plain or gzip-compressed JSONL file in binary mode."""
    if path.suffix == ".gz":
        return cast("IO[bytes]", gzip.open(path, "rb"))
    return path.open("rb")


def iter_records(path: str | Path = DEFAULT_DATA_PATH) -> Iterator[dict[str, Any]]:
    """Yield raw dict records one at a time (streaming, low memory)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Drop the organizer file at "
            f"'{DEFAULT_DATA_PATH}' (or pass an explicit path)."
        )
    with _open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            yield orjson.loads(line)


def iter_candidates(path: str | Path = DEFAULT_DATA_PATH) -> Iterator[Candidate]:
    """Yield parsed ``Candidate`` objects one at a time."""
    for rec in iter_records(path):
        yield Candidate.from_dict(rec)


def count_records(path: str | Path = DEFAULT_DATA_PATH) -> int:
    """Count records without holding them in memory."""
    return sum(1 for _ in iter_records(path))
