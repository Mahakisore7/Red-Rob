"""Retriever tests: RRF fusion + FAISS/BM25 top-k on a tiny index."""

from __future__ import annotations

import numpy as np

from aptus.retriever import bm25_topk, faiss_topk, rrf_merge


def test_rrf_merge_rewards_appearing_in_both_lists() -> None:
    faiss_ids = [10, 20, 30]
    bm25_ids = [30, 40, 50]
    merged = rrf_merge([faiss_ids, bm25_ids], rrf_k=60, pool_size=10)
    # 30 appears in both lists -> highest fused score -> first
    assert merged[0] == 30
    assert set(merged) == {10, 20, 30, 40, 50}


def test_rrf_merge_pool_size_caps() -> None:
    merged = rrf_merge([[1, 2, 3, 4, 5]], rrf_k=60, pool_size=3)
    assert merged == [1, 2, 3]


def test_rrf_merge_deterministic_tiebreak() -> None:
    # two positions with identical fused score -> lower position id first
    merged = rrf_merge([[7], [9]], rrf_k=60, pool_size=10)
    assert merged == [7, 9]


def test_faiss_topk_on_flat_index() -> None:
    import faiss

    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((6, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    # query equal to row 3 -> row 3 should be the top hit
    top = faiss_topk(index, vecs[3], k=3)
    assert top[0] == 3
    assert len(top) == 3


def test_bm25_topk() -> None:
    from rank_bm25 import BM25Okapi

    corpus = [
        ["vector", "search", "ranking"],
        ["payroll", "accounts", "hr"],
        ["embeddings", "retrieval", "ranking"],
    ]
    bm25 = BM25Okapi(corpus)
    top = bm25_topk(bm25, ["ranking", "retrieval"], k=2)
    # docs 0 and 2 are the ranking/retrieval ones; doc 1 (hr) should not lead
    assert 1 not in top[:1]
    assert set(top).issubset({0, 2})
