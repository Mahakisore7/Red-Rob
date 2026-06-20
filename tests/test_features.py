"""Feature precompute tests (S2-S5, modifiers, penalty flags)."""

from __future__ import annotations

from collections.abc import Callable

from aptus import features
from aptus.schema import Candidate


def _cand(record_factory: Callable[..., dict], **_: object) -> Candidate:  # type: ignore[type-arg]
    return Candidate.from_dict(record_factory())


# --- signals --------------------------------------------------------------
def test_signals_in_unit_range(candidate_factory: Callable[..., Candidate]) -> None:
    c = candidate_factory()
    for fn in (features.s3_behavioral, features.s4_recency, features.s5_intent):
        val = fn(c)
        assert 0.0 <= val <= 1.0


def test_s4_recent_vs_dormant(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    rec["redrob_signals"]["last_active_date"] = "2026-05-25"  # ~7 days
    recent = features.s4_recency(Candidate.from_dict(rec))
    rec["redrob_signals"]["last_active_date"] = "2024-06-01"  # ~2 years
    dormant = features.s4_recency(Candidate.from_dict(rec))
    assert recent > 0.9
    assert dormant < recent


def test_s4_bad_date_is_dormant(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    rec["redrob_signals"]["last_active_date"] = "not-a-date"
    assert features.days_inactive(Candidate.from_dict(rec)) == 999


def test_s5_offer_sentinel_neutral(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    rec["redrob_signals"]["offer_acceptance_rate"] = -1
    val = features.s5_intent(Candidate.from_dict(rec))
    assert 0.0 <= val <= 1.0


# --- modifiers ------------------------------------------------------------
def test_notice_mod_buckets(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    def mod(days: int) -> float:
        rec = record_factory()
        rec["redrob_signals"]["notice_period_days"] = days
        return features.notice_mod(Candidate.from_dict(rec))

    assert mod(30) == 1.0
    assert mod(45) == 0.85
    assert mod(90) == 0.70
    assert mod(120) == 0.55


def test_location_mod(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    assert features.location_mod(Candidate.from_dict(rec)) == 1.0  # Pune
    rec["profile"]["location"] = "Jaipur"
    rec["redrob_signals"]["willing_to_relocate"] = True
    assert features.location_mod(Candidate.from_dict(rec)) == 0.85
    rec["redrob_signals"]["willing_to_relocate"] = False
    assert features.location_mod(Candidate.from_dict(rec)) == 0.60
    rec["profile"]["country"] = "USA"
    assert features.location_mod(Candidate.from_dict(rec)) == 0.25


def test_salary_and_work_mod(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    assert features.salary_mod(Candidate.from_dict(rec)) == 1.0  # max 45
    rec["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 70, "max": 90}
    assert features.salary_mod(Candidate.from_dict(rec)) == 0.80
    for mode, expected in (("hybrid", 1.0), ("onsite", 0.90), ("remote", 0.75), ("flexible", 1.0)):
        rec["redrob_signals"]["preferred_work_mode"] = mode
        assert features.work_mod(Candidate.from_dict(rec)) == expected


# --- penalty flags --------------------------------------------------------
def test_consulting_only(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    assert features.is_consulting_only(Candidate.from_dict(rec)) is False
    for entry in rec["career_history"]:
        entry["company"] = "Infosys"
    assert features.is_consulting_only(Candidate.from_dict(rec)) is True


def test_has_product_exp(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    assert features.has_product_exp(Candidate.from_dict(rec)) is True
    rec["profile"]["current_industry"] = "IT Services"
    for entry in rec["career_history"]:
        entry["industry"] = "IT Services"
    assert features.has_product_exp(Candidate.from_dict(rec)) is False


def test_title_chaser_needs_three_short_roles(record_factory: Callable[..., dict]) -> None:  # type: ignore[type-arg]
    rec = record_factory()
    assert features.is_title_chaser(Candidate.from_dict(rec)) is False  # 2 long roles
    short = [
        {
            "company": f"Co{i}",
            "title": "Engineer",
            "start_date": f"20{18 + i}-01-01",
            "end_date": None,
            "duration_months": 10,
            "is_current": i == 0,
            "industry": "Software",
            "company_size": "51-200",
            "description": "",
        }
        for i in range(4)
    ]
    rec["career_history"] = short
    assert features.is_title_chaser(Candidate.from_dict(rec)) is True


# --- S2 aggregation -------------------------------------------------------
def test_aggregate_s2_recency_weighted() -> None:
    assert features.aggregate_s2([]) == 0.0
    assert features.aggregate_s2([0.5]) == 0.5
    # equal scores -> mean equals the score regardless of weights
    assert abs(features.aggregate_s2([1.0, 1.0]) - 1.0) < 1e-9


def test_thesaurus_boost() -> None:
    assert features.thesaurus_boost("built a retrieval and ranking system") > 0.0
    assert features.thesaurus_boost("managed payroll and accounts") == 0.0


# --- assembly -------------------------------------------------------------
def test_build_feature_row(candidate_factory: Callable[..., Candidate]) -> None:
    row = features.build_feature_row(candidate_factory(), s2=0.7, is_honeypot=False)
    d = row.as_dict()
    assert d["candidate_id"] == "CAND_0000001"
    assert 0.0 <= d["s2_career_arc"] <= 1.0
    assert d["has_product_exp"] is True
    assert d["is_honeypot"] is False
