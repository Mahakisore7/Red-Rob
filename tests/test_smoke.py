"""Smoke tests: package imports and CLI entrypoints resolve (Phase 0 setup gate)."""

import aptus
from aptus.cli import eval as eval_cli
from aptus.cli import precompute, rank


def test_package_version():
    assert aptus.__version__ == "0.1.0"


def test_cli_stubs_return_zero():
    # rank/eval are still Phase-2/4 stubs; precompute now requires --candidates.
    assert rank.main([]) == 0
    assert eval_cli.main([]) == 0


def test_precompute_requires_candidates():
    import pytest

    with pytest.raises(SystemExit):
        precompute.main([])


def test_cli_parsers_build():
    assert precompute.build_parser().prog == "aptus-precompute"
    assert rank.build_parser().prog == "aptus-rank"
    assert eval_cli.build_parser().prog == "aptus-eval"
