"""T0.4 — full-pool honeypot gate validation.

Streams the entire candidate dataset through the 6-rule integrity gate
(``aptus.honeypot``), reports per-rule counts, writes ``honeypot_ids.json``,
and (optionally) asserts the documented known-honeypot cases are flagged.

Run
---
    python scripts/honeypot_full_scan.py --candidates data/candidates.jsonl.gz
    python scripts/honeypot_full_scan.py --candidates data/candidates.jsonl.gz --assert-known

Outputs ``artifacts/honeypot_ids.json`` (the same artifact precompute will emit).

Sanity expectation (PHASE_0 §3 / §1 deliverables): the flag count should be a
sane ~70-90 of impossible-profile honeypots, **not** thousands. Note the current
gate also catches the broader keyword-stuffer trap (FR-7e), so the count runs
higher; calibration is a Phase-4 task. ``--assert-known`` fails (exit 1) if any
documented honeypot is *not* flagged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# scripts/ → repo root on path so ``aptus`` + sibling modules import cleanly.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aptus import honeypot  # noqa: E402
from aptus.config import HONEYPOT_CFG  # noqa: E402
from scripts.dataio import DEFAULT_DATA_PATH, iter_candidates  # noqa: E402

logger = logging.getLogger("honeypot_full_scan")

# Documented known honeypots the CURRENT 6-rule gate must catch (PHASE_0 §T0.4 /
# TRD §7). Verify/replace once the official dataset is in hand.
KNOWN_HONEYPOT_IDS: tuple[str, ...] = (
    "CAND_0007353",  # FR-7b career-math mismatch (duration vs stated YoE)
)

# Documented but NOT covered by the current gate: CAND_0004989 is a salary-sanity
# case the spec marks "FR-7 optional" — there is no salary rule implemented yet,
# so it is intentionally excluded from the hard assertion above. Add a salary rule
# (and move this id up) if EDA shows the trap is material.
OPTIONAL_UNCOVERED_IDS: tuple[str, ...] = ("CAND_0004989",)

# Sane flag-count band; outside this we warn (gate may be mis-tuned).
SANE_MIN, SANE_MAX = 40, 250


def scan(candidates_path: str | Path) -> dict[str, object]:
    """Run the gate over the full pool; return a summary dict.

    Returns keys: ``total``, ``flagged_ids`` (list[str]), ``rule_counts`` (dict),
    ``rules_fired_by_id`` (dict[str, list[str]]).
    """
    total = 0
    flagged_ids: list[str] = []
    rule_counts: Counter[str] = Counter()
    rules_by_id: dict[str, list[str]] = {}

    for cand in iter_candidates(candidates_path):
        total += 1
        result = honeypot.check(cand)
        if result.is_honeypot:
            flagged_ids.append(result.candidate_id)
            rules_by_id[result.candidate_id] = result.rules_fired
            for rule in result.rules_fired:
                rule_counts[rule] += 1

    return {
        "total": total,
        "flagged_ids": flagged_ids,
        "rule_counts": dict(sorted(rule_counts.items())),
        "rules_fired_by_id": rules_by_id,
    }


def write_ids(flagged_ids: list[str], out_path: Path) -> None:
    """Write the sorted honeypot id list to ``out_path`` (Phase-A artifact)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sorted(flagged_ids), indent=2), encoding="utf-8")


def _report(summary: dict[str, object]) -> None:
    total = int(summary["total"])  # type: ignore[arg-type]
    flagged = list(summary["flagged_ids"])  # type: ignore[arg-type]
    n = len(flagged)
    pct = 100 * n / total if total else 0.0
    print(f"scanned       : {total:,} candidates")
    print(f"flagged       : {n:,} ({pct:.2f}%)")
    print("per-rule       :")
    for rule, count in summary["rule_counts"].items():  # type: ignore[union-attr]
        print(f"  {rule}: {count:,}")
    if not (SANE_MIN <= n <= SANE_MAX):
        print(
            f"WARNING: flagged count {n} is outside the sane band "
            f"[{SANE_MIN}, {SANE_MAX}] — gate may be mis-tuned.",
            file=sys.stderr,
        )


def _assert_known(summary: dict[str, object]) -> list[str]:
    """Return the list of documented honeypots that were NOT flagged."""
    flagged = set(summary["flagged_ids"])  # type: ignore[arg-type]
    return [cid for cid in KNOWN_HONEYPOT_IDS if cid not in flagged]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(prog="honeypot_full_scan", description=__doc__)
    p.add_argument(
        "--candidates", default=str(DEFAULT_DATA_PATH), help="Path to candidates.jsonl[.gz]."
    )
    p.add_argument(
        "--out",
        default="artifacts/honeypot_ids.json",
        help="Where to write the flagged id list.",
    )
    p.add_argument(
        "--assert-known",
        action="store_true",
        help="Exit non-zero if any documented known honeypot is not flagged.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point: scan, report, write ids, optionally assert known cases."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    summary = scan(args.candidates)
    _report(summary)
    write_ids(list(summary["flagged_ids"]), Path(args.out))  # type: ignore[arg-type]
    mult = HONEYPOT_CFG["multiplier"]
    cap = HONEYPOT_CFG["max_in_top_100"]
    print(f"wrote          : {args.out} (multiplier x{mult}, max_in_top_100={cap})")

    if args.assert_known:
        missing = _assert_known(summary)
        if missing:
            print(f"FAIL: documented honeypots not flagged: {missing}", file=sys.stderr)
            return 1
        print("OK: all documented known honeypots flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
