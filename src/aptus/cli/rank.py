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
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from aptus import signals
from aptus.config import ARTIFACTS_DIR, LLM_CFG, OUTPUT_CFG, RETRIEVAL_CFG
from aptus.jd import build_jd_query
from aptus.llm_reranker import GenerateFn, rerank
from aptus.output_formatter import write_submission
from aptus.retriever import load_artifacts, retrieve
from aptus.scorer import ScoredCandidate, score_pool
from aptus.textproc import tokenize

logger = logging.getLogger(__name__)


def _apply_llm(
    scored: list[ScoredCandidate],
    art_facts: pd.DataFrame,
    generate: GenerateFn,
    weight: float,
    start_time: float,
) -> tuple[list[ScoredCandidate], dict[str, str], int]:
    """Rerank the top-K, blend their scores, and collect LLM reasoning (B7-B8)."""
    top_ids = [s.candidate_id for s in scored][: int(LLM_CFG["default_top_k"])]
    judgments, k_eff = rerank(art_facts, top_ids, generate, start_time=start_time)

    blended: list[ScoredCandidate] = []
    llm_reasoning: dict[str, str] = {}
    for s in scored:
        j = judgments.get(s.candidate_id)
        if j is not None and j.ok and j.fit_score is not None:
            new_score = signals.blend(s.score, j.fit_score, weight)
            blended.append(ScoredCandidate(s.candidate_id, new_score, s.s1))
            if j.reasoning:
                llm_reasoning[s.candidate_id] = j.reasoning
        else:
            blended.append(s)
    return blended, llm_reasoning, k_eff


def run_rank(
    artifacts_dir: str | Path,
    out_path: str | Path,
    *,
    validate: bool = True,
    use_llm: bool = False,
    llm_weight: float | None = None,
    llm_generate: GenerateFn | None = None,
) -> dict[str, object]:
    """Execute the ranking pipeline and write the submission CSV.

    With ``use_llm`` the top-K are reranked by the local LLM and blended; otherwise
    this is the deterministic non-LLM safety path. ``llm_generate`` is injectable for
    tests (defaults to the real Phi-3 generator).
    """
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

    llm_reasoning: dict[str, str] = {}
    summary: dict[str, object] = {}
    if use_llm:
        weight = llm_weight if llm_weight is not None else float(LLM_CFG["default_blend_weight"])
        generate = llm_generate or _default_generator()
        scored, llm_reasoning, k_eff = _apply_llm(scored, art.facts, generate, weight, t0)
        summary["llm_weight"] = weight
        summary["llm_k"] = k_eff
        summary["llm_reasoned"] = len(llm_reasoning)

    summary.update(
        write_submission(
            scored,
            art.facts,
            art.honeypot_ids,
            out_path,
            validate=validate,
            llm_reasoning=llm_reasoning,
        )
    )
    summary["pool_size"] = len(positions)
    summary["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return summary


def _default_generator() -> Callable[[str], str]:  # pragma: no cover - needs the model
    from aptus.llm_reranker import PhiReranker

    return PhiReranker().generate


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aptus-rank`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aptus-rank",
        description="Rank the top 100 candidates (<=5 min, CPU, offline, deterministic).",
    )
    parser.add_argument("--candidates", required=False, help="Unused in the non-LLM path.")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path.")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="Artifacts directory.")
    parser.add_argument("--use-llm", action="store_true", help="Enable Phi-3 rerank of the top-K.")
    parser.add_argument("--llm-weight", type=float, default=None, help="Blend weight w (0-1).")
    parser.add_argument("--top-k", type=int, default=None, help="(reserved) LLM rerank K.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``aptus-rank`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    summary = run_rank(
        args.artifacts_dir, args.out, use_llm=args.use_llm, llm_weight=args.llm_weight
    )
    logger.info("aptus-rank complete: %s", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
