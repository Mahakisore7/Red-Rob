"""``aptus-precompute`` — Phase-A offline artifact builder (TRD §1, PHASE_1 A1-A10).

Builds everything the timed Phase-B step loads:
``faiss.index``, ``bm25.pkl``, ``candidate_features.parquet``,
``candidate_facts.parquet``, ``id_map.json``, ``honeypot_ids.json``,
``jd_embedding.npy``.

The embedder is injected so the pipeline is testable without the heavy model
(``run_precompute`` takes any object exposing ``encode(list[str]) -> ndarray``).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aptus import facts, features, honeypot
from aptus.config import ARTIFACTS_DIR, S2_CFG
from aptus.errors import DataError
from aptus.jd import build_jd_query
from aptus.schema import Candidate, full_text
from aptus.textproc import tokenize

__all__ = ["build_jd_query", "build_parser", "main", "run_precompute"]

logger = logging.getLogger(__name__)

#: Roles considered per candidate for the S2 career-arc signal.
_MAX_S2_ROLES = 5


class SupportsEncode(Protocol):
    """Minimal embedder interface used by the pipeline."""

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts → (N, D) L2-normalized float32."""


def _role_scores(
    candidates: list[Candidate],
    embedder: SupportsEncode,
) -> list[float]:
    """Compute the S2 career-arc scalar for every candidate.

    Embeds each candidate's (up to 5) recent roles, scores each role as the mean
    cosine vs the JD anchors plus a thesaurus boost, then recency-weights them.
    """
    anchor_vecs = embedder.encode(list(S2_CFG["anchors"]))  # (A, D), normalized

    flat_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    boosts: list[float] = []
    for c in candidates:
        start = len(flat_texts)
        for role in features.ordered_roles(c)[:_MAX_S2_ROLES]:
            txt = features.role_text(role)
            flat_texts.append(txt)
            boosts.append(features.thesaurus_boost(txt))
        spans.append((start, len(flat_texts)))

    if flat_texts:
        role_vecs = embedder.encode(flat_texts)  # (R, D)
        cos = role_vecs @ anchor_vecs.T  # (R, A) — normalized => cosine
        per_role = cos.mean(axis=1) + np.asarray(boosts, dtype=np.float32)  # (R,)
    else:  # pragma: no cover - only when no candidate has any role
        per_role = np.zeros((0,), dtype=np.float32)

    s2_values: list[float] = []
    for start, end in spans:
        scores = [float(per_role[i]) for i in range(start, end)]
        s2_values.append(features.aggregate_s2(scores))
    return s2_values


def run_precompute(
    records: Iterable[dict],  # type: ignore[type-arg]
    out_dir: Path,
    embedder: SupportsEncode,
    limit: int | None = None,
) -> dict[str, int]:
    """Run A1-A10 and write all artifacts to ``out_dir``. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # A1 parse + id_map
    candidates: list[Candidate] = []
    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break
        candidates.append(Candidate.from_dict(rec))
    n = len(candidates)
    if n == 0:
        raise DataError("no candidates parsed; check the --candidates path")
    id_map = {i: c.candidate_id for i, c in enumerate(candidates)}
    logger.info("A1 parsed %d candidates", n)

    # A3 honeypot gate
    honeypot_ids = [r.candidate_id for r in honeypot.scan(candidates)]
    hp_set = set(honeypot_ids)
    logger.info("A3 flagged %d honeypots", len(honeypot_ids))

    # A4 text + A5 embed candidates
    texts = [full_text(c) for c in candidates]
    cand_vecs = embedder.encode(texts).astype(np.float32)
    dim = cand_vecs.shape[1]
    logger.info("A5 embedded candidates -> %s", cand_vecs.shape)

    # A6 FAISS IndexFlatIP (cosine on normalized vectors)
    import faiss

    index = faiss.IndexFlatIP(dim)
    index.add(cand_vecs)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    logger.info("A6 wrote faiss.index (ntotal=%d)", index.ntotal)

    # A7 BM25 over the same texts
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with (out_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump({"bm25": bm25, "id_map": id_map}, fh)
    logger.info("A7 wrote bm25.pkl")

    # A8 features (S2 from embeddings + S3-S5/mods/flags arithmetic)
    s2_values = _role_scores(candidates, embedder)
    feat_rows = [
        features.build_feature_row(c, s2_values[i], c.candidate_id in hp_set).as_dict()
        for i, c in enumerate(candidates)
    ]
    pd.DataFrame(feat_rows).to_parquet(out_dir / "candidate_features.parquet", index=False)
    logger.info("A8 wrote candidate_features.parquet")

    # A9 facts
    fact_rows = [facts.build_fact_row(c).as_dict() for c in candidates]
    pd.DataFrame(fact_rows).to_parquet(out_dir / "candidate_facts.parquet", index=False)
    logger.info("A9 wrote candidate_facts.parquet")

    # A10 JD embedding
    jd_vec = embedder.encode([build_jd_query()]).astype(np.float32)[0]
    np.save(out_dir / "jd_embedding.npy", jd_vec)
    logger.info("A10 wrote jd_embedding.npy")

    # id_map + honeypot ids
    (out_dir / "id_map.json").write_text(
        json.dumps({str(k): v for k, v in id_map.items()}), encoding="utf-8"
    )
    (out_dir / "honeypot_ids.json").write_text(
        json.dumps(sorted(honeypot_ids), indent=2), encoding="utf-8"
    )

    # Invariant: every artifact length == n (fail fast in Phase A, never Phase B)
    if not (len(feat_rows) == len(fact_rows) == len(id_map) == n):
        raise DataError(
            f"length invariant violated: features={len(feat_rows)} facts={len(fact_rows)} "
            f"id_map={len(id_map)} n={n}"
        )

    return {"n": n, "honeypots": len(honeypot_ids), "dim": dim}


def _iter_records(path: str) -> Iterator[dict]:  # type: ignore[type-arg] # pragma: no cover
    # Thin wrapper kept out of coverage; the loader itself is tested via scripts.dataio.
    from scripts.dataio import iter_records

    yield from iter_records(path)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aptus-precompute`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aptus-precompute",
        description="Build Phase-A artifacts (embeddings, FAISS, BM25, features, facts).",
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz].")
    parser.add_argument("--out-dir", default=str(ARTIFACTS_DIR), help="Artifact output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records.")
    parser.add_argument("--device", default="cpu", help="Embedder device (cpu/cuda).")
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - integration entry point
    """Entry point for the ``aptus-precompute`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    from aptus.embedder import Embedder

    embedder = Embedder(device=args.device)
    summary = run_precompute(
        _iter_records(args.candidates), Path(args.out_dir), embedder, limit=args.limit
    )
    logger.info("precompute complete: %s", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
