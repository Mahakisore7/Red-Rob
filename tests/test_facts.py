"""Reasoning-facts precompute tests."""

from __future__ import annotations

from collections.abc import Callable

from aptus import facts
from aptus.schema import Candidate


def test_top_skills_sorted_by_duration(candidate_factory: Callable[..., Candidate]) -> None:
    s = facts.top_skills_str(candidate_factory())
    # Python has the longest duration (84mo) -> appears first
    assert s.startswith("Python (expert, 84mo)")
    assert "PyTorch" in s and "FAISS" in s


def test_gaps_empty_for_clean_candidate(candidate_factory: Callable[..., Candidate]) -> None:
    assert facts.gaps_str(candidate_factory()) == ""


def test_gaps_flags_long_notice(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    rec["redrob_signals"]["notice_period_days"] = 120
    gaps = facts.gaps_str(Candidate.from_dict(rec))
    assert "120-day notice" in gaps


def test_gaps_flags_dormant_and_no_product(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    rec["redrob_signals"]["last_active_date"] = "2024-01-01"
    rec["profile"]["current_industry"] = "IT Services"
    for entry in rec["career_history"]:
        entry["industry"] = "IT Services"
    gaps = facts.gaps_str(Candidate.from_dict(rec))
    assert "dormant" in gaps
    assert "no product-company experience" in gaps


def test_build_fact_row(candidate_factory: Callable[..., Candidate]) -> None:
    row = facts.build_fact_row(candidate_factory()).as_dict()
    assert row["candidate_id"] == "CAND_0000001"
    assert row["current_title"] == "Senior ML Engineer"
    assert row["years_of_experience"] == 7.0
    assert row["notice_period_days"] == 30
    assert isinstance(row["gaps"], str)
