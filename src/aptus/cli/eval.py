"""``aptus-eval`` -- Phase C untimed evaluation (TRD section 2 FR-24/25, PHASE_4).

Phase 0 stub: argument surface only. The real harness (NDCG@10/@50, MAP, P@10,
honeypot rate + baseline ablation) lands in Phase 4.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aptus-eval`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aptus-eval",
        description="Evaluate a submission against the weak ground-truth gold set.",
    )
    parser.add_argument("--submission", required=False, help="Path to submission.csv.")
    parser.add_argument("--gold", default="eval/gold_set.csv", help="Path to gold_set.csv.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``aptus-eval`` console script."""
    args = build_parser().parse_args(argv)
    logger.info("aptus-eval stub invoked (gold=%s); implemented in Phase 4.", args.gold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
