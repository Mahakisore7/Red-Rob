"""Tokenizer tests (BM25 indexing/query parity)."""

from __future__ import annotations

from aptus.textproc import tokenize


def test_basic_split_and_lowercase() -> None:
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_splits_on_punctuation_and_underscore() -> None:
    assert tokenize("a-b_c.d") == ["a", "b", "c", "d"]


def test_empty_and_symbols_only() -> None:
    assert tokenize("") == []
    assert tokenize("!!! ??? ...") == []


def test_keeps_alphanumerics() -> None:
    assert tokenize("vector-search BM25 NDCG@10") == ["vector", "search", "bm25", "ndcg", "10"]
