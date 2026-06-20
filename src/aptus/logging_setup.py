"""Structured logging setup (structlog) for Aptus-R (docs/07 §5).

A single ``configure_logging`` entry point used by the CLIs. Console-friendly
key=value rendering, ISO timestamps, level filtering. Deterministic and
side-effect-free beyond configuring the global structlog/logging state.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging once, at process start.

    Args:
        level: Minimum level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=numeric_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for ``name``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
