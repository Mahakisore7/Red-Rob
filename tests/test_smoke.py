"""Smoke tests: package imports and CLI entrypoints resolve (Phase 0 setup gate)."""

import aptus
from aptus.cli import eval as eval_cli
from aptus.cli import precompute, rank


def test_package_version():
    assert aptus.__version__ == "0.1.0"


def test_cli_stubs_return_zero():
    assert precompute.main([]) == 0
    assert rank.main([]) == 0
    assert eval_cli.main([]) == 0


def test_cli_parsers_build():
    assert precompute.build_parser().prog == "aptus-precompute"
    assert rank.build_parser().prog == "aptus-rank"
    assert eval_cli.build_parser().prog == "aptus-eval"
