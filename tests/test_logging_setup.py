"""Logging setup smoke tests."""

from __future__ import annotations

from aptus.logging_setup import configure_logging, get_logger


def test_configure_and_get_logger() -> None:
    configure_logging("DEBUG")
    log = get_logger("test")
    # Should be a usable bound logger; calling it must not raise.
    log.info("hello", key="value")
    configure_logging("INFO")  # reset to default level
