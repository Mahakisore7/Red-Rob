"""``aptus-rank`` -- Phase B timed ranking step (TRD section 1, PHASE_2/PHASE_3).

Phase 0 stub: argument surface only. The real pipeline (load artifacts -> retrieve
-> score -> LLM rerank -> reason -> write CSV -> self-validate) lands in Phase 2-3.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aptus-rank`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aptus-rank",
        description="Rank the top 100 candidates (<=5 min, CPU, offline, deterministic).",
    )
    parser.add_argument("--candidates", required=False, help="Path to candidates.jsonl[.gz].")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path.")
    parser.add_argument("--llm-weight", type=float, default=None, help="Override blend weight w.")
    parser.add_argument("--top-k", type=int, default=None, help="Override LLM rerank K.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``aptus-rank`` console script."""
    args = build_parser().parse_args(argv)
    logger.info("aptus-rank stub invoked (out=%s); implemented in Phase 2-3.", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
