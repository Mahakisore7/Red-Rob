"""bge-large-en-v1.5 embedding wrapper (TRD FR-5, PHASE_1 A5).

Phase-A only. The timed Phase-B ranking path imports precomputed vectors, never
this module (the heavy ``sentence_transformers`` / ``torch`` stack lives in the
``precompute`` optional extra).

Embeddings are L2-normalized so FAISS ``IndexFlatIP`` == cosine similarity.
Identical texts are embedded once (dedup cache) — cheap insurance on a dataset
with heavily templated role descriptions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from aptus.errors import ModelError

#: Alias for the float32 embedding arrays this module produces/consumes.
F32Array = NDArray[np.float32]

#: bge-large-en-v1.5 output dimensionality.
EMBED_DIM: int = 1024

#: Default local embedding model (cached under ``models/`` on first use).
DEFAULT_MODEL: str = "BAAI/bge-large-en-v1.5"


def dedup_encode(
    texts: Sequence[str],
    encode_fn: Callable[[list[str]], F32Array],
) -> F32Array:
    """Embed ``texts`` calling ``encode_fn`` only on the unique strings.

    Pure, model-free orchestration (unit-tested): de-duplicates inputs, embeds the
    distinct set via ``encode_fn``, then scatters rows back to the original order.

    Args:
        texts: Texts to embed (order preserved in the output).
        encode_fn: Callable mapping a list of unique texts to an ``(U, D)`` array.

    Returns:
        ``(len(texts), D)`` float32 array, row ``i`` = embedding of ``texts[i]``.
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    unique: dict[str, int] = {}
    for t in texts:
        if t not in unique:
            unique[t] = len(unique)

    uniq_list = list(unique.keys())
    uniq_vecs = encode_fn(uniq_list)
    if uniq_vecs.shape[0] != len(uniq_list):
        raise ModelError(f"encode_fn returned {uniq_vecs.shape[0]} rows for {len(uniq_list)} texts")

    out = np.empty((n, uniq_vecs.shape[1]), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = uniq_vecs[unique[t]]
    return out


class Embedder:
    """Lazy bge-large wrapper. Loads the model on first ``encode`` call."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_folder: str | None = "models",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        """Store config; defer the (heavy) model load until first use."""
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self.batch_size = batch_size
        self._model: object | None = None

    def _load(self) -> None:  # pragma: no cover - exercised only with the real model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ModelError(
                "sentence-transformers not installed; install the 'precompute' extra"
            ) from exc
        self._model = SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, device=self.device
        )

    def _encode_raw(self, texts: list[str]) -> F32Array:  # pragma: no cover - real model
        if self._model is None:
            self._load()
        vecs = self._model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 5000,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode(self, texts: Sequence[str]) -> F32Array:
        """Embed ``texts`` → ``(N, EMBED_DIM)`` L2-normalized float32, dedup-cached."""
        return dedup_encode(texts, self._encode_raw)
