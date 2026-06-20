"""Text tokenization for BM25 retrieval (TRD FR-8).

Shared by Phase-A BM25 index construction (A7) and the Phase-B query path so the
*same* tokenization is used for indexing and querying. Deterministic, stdlib-only.
"""

from __future__ import annotations

import re

# Split on any run of non-alphanumeric characters (keeps digits, drops punctuation).
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split into alphanumeric tokens.

    Empty tokens are dropped. Used identically for BM25 indexing and querying so
    candidate texts and the JD query share a vocabulary.

    Args:
        text: Raw text (candidate blob or JD query).

    Returns:
        List of lowercased alphanumeric tokens (possibly empty).
    """
    return [tok for tok in _TOKEN_SPLIT.split(text.lower()) if tok]
