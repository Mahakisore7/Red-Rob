"""Shared streaming data loader for analysis scripts / notebooks.

Thin re-export of the canonical loader in ``aptus.dataio`` (kept here so the
notebook and operational scripts can ``from scripts.dataio import ...``). Adds a
sys.path shim so it also works from a bare ``python scripts/...`` invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``import aptus`` work when run as a bare script (no installed package).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aptus.dataio import (  # noqa: E402  (re-export)
    DEFAULT_DATA_PATH,
    count_records,
    iter_candidates,
    iter_records,
)

__all__ = ["DEFAULT_DATA_PATH", "count_records", "iter_candidates", "iter_records"]
