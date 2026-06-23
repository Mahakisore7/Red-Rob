"""``aptus-rank`` — Phase-B timed ranking step (TRD §1, PHASE_2 B1-B10).

Non-LLM path (the safety submission): load artifacts → retrieve (FAISS+BM25+RRF)
→ 5-signal score → grounded template reasoning → write CSV → self-validate.
Imports no embedder/torch and makes no network calls — it reads the precomputed
``jd_embedding.npy`` rather than re-embedding the JD.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from aptus.config import ARTIFACTS_DIR, OUTPUT_CFG, RETRIEVAL_CFG
from aptus.jd import build_jd_query
from aptus.output_formatter import write_submission
from aptus.retriever import load_artifacts, retrieve
from aptus.scorer import score_pool
from aptus.textproc import tokenize

logger = logging.getLogger(__name__)


def run_rank(
    artifacts_dir: str | Path,
    out_path: str | Path,
    *,
    validate: bool = True,
) -> dict[str, object]:
    """Execute the non-LLM ranking pipeline and write the submission CSV."""
    t0 = time.perf_counter()
    art = load_artifacts(artifacts_dir)  # B1
    jd_tokens = tokenize(build_jd_query())  # B2 (semantic side uses precomputed jd vector)
    positions = retrieve(  # B3-B5
        art.jd_vector,
        jd_tokens,
        art.index,
        art.bm25,
        faiss_k=int(RETRIEVAL_CFG["faiss_top_k"]),
        bm25_k=int(RETRIEVAL_CFG["bm25_top_k"]),
        rrf_k=int(RETRIEVAL_CFG["rrf_k"]),
        pool_size=int(RETRIEVAL_CFG["pool_size"]),
    )
    scored = score_pool(art, positions, int(OUTPUT_CFG["n_results"]))  # B6
    summary = write_submission(scored, art.facts, art.honeypot_ids, out_path, validate=validate)
    summary["pool_size"] = len(positions)
    summary["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aptus-rank`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aptus-rank",
        description="Rank the top 100 candidates (<=5 min, CPU, offline, deterministic).",
    )
    parser.add_argument("--candidates", required=False, help="Unused in the non-LLM path.")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path.")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="Artifacts directory.")
    parser.add_argument("--llm-weight", type=float, default=None, help="(Phase 3) blend weight w.")
    parser.add_argument("--top-k", type=int, default=None, help="(Phase 3) LLM rerank K.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``aptus-rank`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    summary = run_rank(args.artifacts_dir, args.out)
    logger.info("aptus-rank complete: %s", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
