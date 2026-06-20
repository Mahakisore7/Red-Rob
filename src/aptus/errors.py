"""Typed exception hierarchy for Aptus-R (docs/07 §6).

A small, explicit set of errors so callers can distinguish *why* something failed
(bad data vs missing artifact vs model load) and fail loud where the spec demands
it (e.g. Phase-A invariant violations).
"""

from __future__ import annotations


class AptusError(Exception):
    """Base class for all Aptus-R errors."""


class DataError(AptusError):
    """Raised when input data is malformed or violates an expected invariant."""


class ArtifactError(AptusError):
    """Raised when a precomputed artifact is missing, mismatched, or corrupt."""


class ModelError(AptusError):
    """Raised when a local model (embedder / LLM) cannot be loaded or run."""


class ConfigError(AptusError):
    """Raised when configuration is missing a required key or holds a bad value."""
