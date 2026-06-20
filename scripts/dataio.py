"""Shared streaming data loader for Phase-0 EDA + honeypot validation.

Lives under ``scripts/`` (not ``src/aptus``) so it stays outside the package
coverage gate — these are operational/analysis helpers, not shipped runtime code.

Stream-parses ``candidates.jsonl[.gz]`` one record at a time (TRD NFR-2: never
hold all raw JSON in memory). Reuses the already-tested ``aptus.schema`` parser.

Usage
-----
    from scripts.dataio import iter_candidates
    for cand in iter_candidates("data/candidates.jsonl.gz"):
        ...
"""

from __future__ import annotations

import gzip
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO, cast

# Make ``import aptus`` work when this file is imported from a notebook or a
# bare ``python scripts/...`` invocation without an installed package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import orjson  # noqa: E402  (after sys.path shim)

from aptus.schema import Candidate  # noqa: E402

# The organizer ships a plain .jsonl; the loader also transparently handles .gz.
DEFAULT_DATA_PATH = Path("data/candidates.jsonl")


def _open(path: Path) -> IO[bytes]:
    """Open a plain or gzip-compressed JSONL file in binary mode."""
    if path.suffix == ".gz":
        return cast("IO[bytes]", gzip.open(path, "rb"))
    return path.open("rb")


def iter_records(path: str | Path = DEFAULT_DATA_PATH) -> Iterator[dict]:  # type: ignore[type-arg]
    """Yield raw dict records one at a time (streaming, low memory)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Drop the organizer file at "
            f"'{DEFAULT_DATA_PATH}' (or pass an explicit path)."
        )
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
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
