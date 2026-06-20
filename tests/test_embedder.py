"""Embedder dedup/scatter logic tests (no model required)."""

from __future__ import annotations

import numpy as np

from aptus.embedder import EMBED_DIM, dedup_encode


def test_dedup_encode_calls_once_and_scatters() -> None:
    calls: list[list[str]] = []

    def fake_encode(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.array([[float(ord(t[0]))] * 4 for t in texts], dtype=np.float32)

    out = dedup_encode(["a", "b", "a", "c"], fake_encode)

    # encode_fn called exactly once, on the 3 unique texts
    assert len(calls) == 1
    assert calls[0] == ["a", "b", "c"]
    # output preserves input order and length
    assert out.shape == (4, 4)
    # duplicate "a" rows are identical
    assert np.array_equal(out[0], out[2])
    assert out[0][0] == float(ord("a"))


def test_dedup_encode_empty() -> None:
    def fake_encode(texts: list[str]) -> np.ndarray:  # pragma: no cover - not called
        return np.zeros((len(texts), EMBED_DIM), dtype=np.float32)

    out = dedup_encode([], fake_encode)
    assert out.shape == (0, EMBED_DIM)
