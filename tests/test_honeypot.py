"""Honeypot gate unit tests -- one synthetic candidate per rule (TRD §7).

Each test builds the minimal dict that triggers exactly one rule, confirms
the rule fires, and also confirms a clean candidate does NOT fire.
"""

from __future__ import annotations

from aptus.honeypot import check
from aptus.schema import Candidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_dict(candidate_id: str = "TEST_0000001") -> dict:  # type: ignore[type-arg]
    """Minimal valid candidate dict that passes all 6 rules."""
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Candidate A",
            "headline": "Software Engineer",
            "summary": "Clean candidate for tests.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "Senior Software Engineer",
            "current_company": "Acme Corp",
            "current_company_size": "500-1000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Software Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Technology",
                "company_size": "500-1000",
                "description": "Built backend services.",
            }
        ],
        "education": [],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 36},
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 85,
            "signup_date": "2023-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 200,
            "endorsements_received": 10,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.9,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def _candidate(d: dict) -> Candidate:  # type: ignore[type-arg]
    return Candidate.from_dict(d)


# ---------------------------------------------------------------------------
# Baseline: clean candidate must NOT be flagged
# ---------------------------------------------------------------------------


def test_clean_candidate_not_flagged() -> None:
    result = check(_candidate(_base_dict("CLEAN_001")))
    assert not result.is_honeypot, f"Clean candidate flagged unexpectedly: {result.rules_fired}"


# ---------------------------------------------------------------------------
# FR-7a: expert + zero duration on >=2 skills
# ---------------------------------------------------------------------------


def test_fr7a_expert_zero_duration() -> None:
    d = _base_dict("FR7A_001")
    d["skills"] = [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "TensorFlow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Python", "proficiency": "intermediate", "endorsements": 5, "duration_months": 24},
    ]
    result = check(_candidate(d))
    assert result.is_honeypot
    assert "FR-7a" in result.rules_fired


def test_fr7a_one_expert_zero_not_flagged() -> None:
    """Only 1 expert+zero skill should NOT trigger FR-7a (threshold is 2)."""
    d = _base_dict("FR7A_002")
    d["skills"] = [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 36},
    ]
    result = check(_candidate(d))
    assert "FR-7a" not in result.rules_fired


# ---------------------------------------------------------------------------
# FR-7b: career math mismatch
# ---------------------------------------------------------------------------


def test_fr7b_career_math_mismatch() -> None:
    d = _base_dict("FR7B_001")
    d["profile"]["years_of_experience"] = 3.0  # => 3*12=36 months + 24 slack = 60
    d["career_history"] = [
        {
            "company": "A",
            "title": "Eng",
            "start_date": "2010-01-01",
            "end_date": None,
            "duration_months": 61,  # 61 > 60 => fires
            "is_current": True,
            "industry": "Tech",
            "company_size": "50",
            "description": "",
        }
    ]
    result = check(_candidate(d))
    assert result.is_honeypot
    assert "FR-7b" in result.rules_fired


def test_fr7b_within_slack_not_flagged() -> None:
    d = _base_dict("FR7B_002")
    d["profile"]["years_of_experience"] = 5.0  # 60 months + 24 slack = 84 allowed
    d["career_history"] = [
        {
            "company": "A",
            "title": "Eng",
            "start_date": "2019-01-01",
            "end_date": None,
            "duration_months": 60,
            "is_current": True,
            "industry": "Tech",
            "company_size": "50",
            "description": "",
        }
    ]
    result = check(_candidate(d))
    assert "FR-7b" not in result.rules_fired


# ---------------------------------------------------------------------------
# FR-7c: too many experts (>8)
# ---------------------------------------------------------------------------


def test_fr7c_too_many_experts() -> None:
    d = _base_dict("FR7C_001")
    d["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 1, "duration_months": 12}
        for i in range(9)  # 9 > 8 threshold
    ]
    result = check(_candidate(d))
    assert result.is_honeypot
    assert "FR-7c" in result.rules_fired


def test_fr7c_exactly_eight_not_flagged() -> None:
    d = _base_dict("FR7C_002")
    d["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 1, "duration_months": 12}
        for i in range(8)  # exactly 8, threshold is >8
    ]
    result = check(_candidate(d))
    assert "FR-7c" not in result.rules_fired


# ---------------------------------------------------------------------------
# FR-7d: perfect completeness, no verification
# ---------------------------------------------------------------------------


def test_fr7d_perfect_no_verify() -> None:
    d = _base_dict("FR7D_001")
    d["redrob_signals"]["profile_completeness_score"] = 100
    d["redrob_signals"]["verified_email"] = False
    d["redrob_signals"]["verified_phone"] = False
    result = check(_candidate(d))
    assert result.is_honeypot
    assert "FR-7d" in result.rules_fired


def test_fr7d_perfect_with_email_verify_not_flagged() -> None:
    d = _base_dict("FR7D_002")
    d["redrob_signals"]["profile_completeness_score"] = 100
    d["redrob_signals"]["verified_email"] = True  # one verification is enough
    d["redrob_signals"]["verified_phone"] = False
    result = check(_candidate(d))
    assert "FR-7d" not in result.rules_fired


# ---------------------------------------------------------------------------
# FR-7e: keyword stuffer (non-tech title + >=8 AI/ML skills at adv/expert)
# ---------------------------------------------------------------------------


def test_fr7e_keyword_stuffer() -> None:
    d = _base_dict("FR7E_001")
    d["profile"]["current_title"] = "HR Manager"  # non-technical
    d["skills"] = [
        {
            "name": f"Machine Learning Skill {i}",
            "proficiency": "expert",
            "endorsements": 0,
            "duration_months": 0,
        }
        for i in range(8)
    ]
    result = check(_candidate(d))
    assert result.is_honeypot
    assert "FR-7e" in result.rules_fired


def test_fr7e_tech_title_not_flagged() -> None:
    """A technical title with many ML skills is NOT a keyword stuffer."""
    d = _base_dict("FR7E_002")
    d["profile"]["current_title"] = "ML Engineer"  # technical
    d["skills"] = [
        {
            "name": f"Machine Learning Skill {i}",
            "proficiency": "expert",
            "endorsements": 0,
            "duration_months": 0,
        }
        for i in range(8)
    ]
    result = check(_candidate(d))
    assert "FR-7e" not in result.rules_fired


# ---------------------------------------------------------------------------
# FR-7f: assessment contradiction (expert claim but score < 30)
# ---------------------------------------------------------------------------


def test_fr7f_assessment_contradiction() -> None:
    d = _base_dict("FR7F_001")
    d["skills"] = [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 5, "duration_months": 24},
    ]
    d["redrob_signals"]["skill_assessment_scores"] = {"PyTorch": 15}  # expert but scored 15
    result = check(_candidate(d))
    assert result.is_honeypot
    assert "FR-7f" in result.rules_fired


def test_fr7f_high_assessment_not_flagged() -> None:
    d = _base_dict("FR7F_002")
    d["skills"] = [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 5, "duration_months": 24},
    ]
    d["redrob_signals"]["skill_assessment_scores"] = {"PyTorch": 85}  # expert + high score = ok
    result = check(_candidate(d))
    assert "FR-7f" not in result.rules_fired


def test_fr7f_no_assessment_not_flagged() -> None:
    """expert with NO assessment entry at all should not trigger FR-7f."""
    d = _base_dict("FR7F_003")
    d["skills"] = [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 5, "duration_months": 24},
    ]
    d["redrob_signals"]["skill_assessment_scores"] = {}
    result = check(_candidate(d))
    assert "FR-7f" not in result.rules_fired
