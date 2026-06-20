"""T0.4 — full-pool honeypot scan integration test.

These run only when the organizer dataset is present at
``data/candidates.jsonl.gz`` (it is gitignored and absent in CI), otherwise they
skip. The per-rule unit tests in ``test_honeypot.py`` provide the always-on
coverage; this file validates the gate against the *real* pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.honeypot_full_scan import (  # noqa: E402
    KNOWN_HONEYPOT_IDS,
    SANE_MAX,
    SANE_MIN,
    scan,
)

_DATA = _ROOT / "data" / "candidates.jsonl"

pytestmark = pytest.mark.skipif(
    not _DATA.exists(),
    reason="dataset not present (data/candidates.jsonl.gz); skipping full-pool scan",
)


@pytest.fixture(scope="module")
def summary() -> dict:  # type: ignore[type-arg]
    return scan(_DATA)


def test_full_pool_parsed(summary: dict) -> None:  # type: ignore[type-arg]
    assert summary["total"] > 0


def test_flag_count_in_sane_band(summary: dict) -> None:  # type: ignore[type-arg]
    n = len(summary["flagged_ids"])
    assert SANE_MIN <= n <= SANE_MAX, f"flag count {n} outside sane band [{SANE_MIN}, {SANE_MAX}]"


def test_known_honeypots_flagged(summary: dict) -> None:  # type: ignore[type-arg]
    flagged = set(summary["flagged_ids"])
    missing = [cid for cid in KNOWN_HONEYPOT_IDS if cid not in flagged]
    assert not missing, f"documented honeypots not flagged: {missing}"
