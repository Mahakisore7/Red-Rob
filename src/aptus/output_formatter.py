"""Submission CSV writer + self-validation (TRD FR-20/21/22/23, PHASE_2 B10).

Guarantees the exact contract the organizer validator enforces: 100 rows,
header order, scores rounded + non-increasing, ties broken by candidate_id
ascending. Then runs the vendored ``validate_submission.py`` and asserts the
top-100 honeypot count is within budget.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from aptus import reasoning
from aptus.config import HONEYPOT_CFG, OUTPUT_CFG, REPO_ROOT
from aptus.errors import DataError
from aptus.scorer import ScoredCandidate

_HEADER: list[str] = list(OUTPUT_CFG["columns"])
_DECIMALS = int(OUTPUT_CFG["score_decimals"])
_N_RESULTS = int(OUTPUT_CFG["n_results"])

Row = tuple[str, int, str, str]  # candidate_id, rank, score_str, reasoning


def build_rows(
    scored: Sequence[ScoredCandidate],
    facts: pd.DataFrame,
    llm_reasoning: Mapping[str, str] | None = None,
) -> list[Row]:
    """Round scores, enforce tie-break + non-increasing, attach grounded reasoning.

    ``llm_reasoning`` maps candidate_id -> raw LLM reasoning; it is used only when it
    passes the grounding validator, otherwise the template is used (FR-18/19).
    """
    llm_reasoning = llm_reasoning or {}
    rounded = sorted(
        ((round(s.score, _DECIMALS), s.candidate_id) for s in scored),
        key=lambda x: (-x[0], x[1]),  # score desc, candidate_id asc (validator tie-break)
    )
    rows: list[Row] = []
    prev: float | None = None
    for rank, (score, cid) in enumerate(rounded, start=1):
        if prev is not None and score > prev:  # defensive: keep non-increasing
            score = prev
        prev = score
        fact = cast("dict[str, object]", facts.loc[cid].to_dict())
        fact["candidate_id"] = cid
        reason = reasoning.choose_reasoning(fact, rank, llm_reasoning.get(cid))
        rows.append((cid, rank, f"{score:.{_DECIMALS}f}", reason))
    return rows


def write_csv(rows: Sequence[Row], out_path: str | Path) -> None:
    """Write the submission CSV (csv.writer auto-quotes commas in reasoning)."""
    path = Path(out_path)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)
        for cid, rank, score, reason in rows:
            writer.writerow([cid, rank, score, reason])


def run_validator(out_path: str | Path) -> tuple[int, str]:
    """Run the vendored validate_submission.py; return (returncode, output)."""
    script = REPO_ROOT / "validate_submission.py"
    result = subprocess.run(
        [sys.executable, str(script), str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def assert_honeypots(rows: Sequence[Row], honeypot_ids: set[str]) -> int:
    """Assert the top-100 honeypot count is within budget; return the count."""
    count = sum(1 for cid, *_ in rows if cid in honeypot_ids)
    cap = int(HONEYPOT_CFG["max_in_top_100"])
    if count > cap:
        raise DataError(f"too many honeypots in top-100: {count} > {cap}")
    return count


def write_submission(
    scored: Sequence[ScoredCandidate],
    facts: pd.DataFrame,
    honeypot_ids: set[str],
    out_path: str | Path,
    *,
    validate: bool = True,
    llm_reasoning: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build rows, write CSV, assert honeypots, and (optionally) run the validator."""
    rows = build_rows(scored, facts, llm_reasoning)
    if len(rows) != _N_RESULTS:
        raise DataError(f"expected {_N_RESULTS} rows, got {len(rows)}")
    write_csv(rows, out_path)
    honeypots = assert_honeypots(rows, honeypot_ids)

    summary: dict[str, object] = {"rows": len(rows), "honeypots_in_top100": honeypots}
    if validate:
        code, output = run_validator(out_path)
        if code != 0:
            raise DataError(f"validate_submission.py failed (code {code}):\n{output}")
        summary["validator"] = "passed"
    return summary
