"""``aptus-precompute`` -- Phase A offline artifact builder (TRD section 1, PHASE_1).

Phase 0 stub: argument surface only. The real pipeline (parse -> skill_index ->
honeypot gate -> embed -> FAISS/BM25 -> features/facts) lands in Phase 1.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aptus-precompute`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aptus-precompute",
        description="Build Phase-A artifacts (embeddings, FAISS, BM25, features).",
    )
    parser.add_argument("--candidates", required=False, help="Path to candidates.jsonl[.gz].")
    parser.add_argument("--out-dir", default="artifacts/", help="Artifact output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``aptus-precompute`` console script."""
    args = build_parser().parse_args(argv)
    logger.info("aptus-precompute stub invoked (out_dir=%s); implemented in Phase 1.", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
