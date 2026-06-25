"""Smoke tests: package imports and CLI entrypoints resolve (Phase 0 setup gate)."""

import aptus
from aptus.cli import eval as eval_cli
from aptus.cli import precompute, rank


def test_package_version():
    assert aptus.__version__ == "0.1.0"


def test_eval_requires_submission():
    import pytest

    with pytest.raises(SystemExit):
        eval_cli.main([])


def test_precompute_requires_candidates():
    import pytest

    with pytest.raises(SystemExit):
        precompute.main([])


def test_rank_requires_artifacts(tmp_path):
    # rank now loads Phase-A artifacts; missing dir -> ArtifactError.
    import pytest

    from aptus.errors import ArtifactError

    with pytest.raises(ArtifactError):
        rank.main(["--artifacts-dir", str(tmp_path / "nope")])


def test_cli_parsers_build():
    assert precompute.build_parser().prog == "aptus-precompute"
    assert rank.build_parser().prog == "aptus-rank"
    assert eval_cli.build_parser().prog == "aptus-eval"
