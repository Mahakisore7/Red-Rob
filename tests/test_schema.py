"""Tests for schema.py -- Candidate parsing, skill_index(), full_text()."""

from __future__ import annotations

from aptus.honeypot import scan
from aptus.schema import Candidate, full_text, skill_index


def _base_dict(cid: str = "SCHEMA_001") -> dict:  # type: ignore[type-arg]
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test User",
            "headline": "AI Engineer at Startup",
            "summary": "Building retrieval systems.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Senior ML Engineer",
            "current_company": "Startup AI",
            "current_company_size": "50-200",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "Startup AI",
                "title": "ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 36,
                "is_current": True,
                "industry": "Technology",
                "company_size": "50-200",
                "description": "Built embedding pipelines and FAISS indexes.",
            },
            {
                "company": "Old Corp",
                "title": "Data Scientist",
                "start_date": "2018-01-01",
                "end_date": "2021-01-01",
                "duration_months": 36,
                "is_current": False,
                "industry": "Finance",
                "company_size": "1000+",
                "description": "Trained ranking models.",
            },
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "9.0",
                "tier": "Tier-1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 60},
            {
                "name": "PyTorch",
                "proficiency": "advanced",
                "endorsements": 10,
                "duration_months": 30,
            },
            {
                "name": "FAISS",
                "proficiency": "intermediate",
                "endorsements": 5,
                "duration_months": 12,
            },
        ],
        "certifications": [{"name": "AWS ML Specialty"}],
        "languages": [{"name": "English", "proficiency": "fluent"}],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2022-01-01",
            "last_active_date": "2026-05-15",
            "open_to_work_flag": True,
            "profile_views_received_30d": 25,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 6.0,
            "skill_assessment_scores": {"Python": 88, "NumPy": 75},
            "connection_count": 500,
            "endorsements_received": 35,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 55},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 70,
            "search_appearance_30d": 80,
            "saved_by_recruiters_30d": 12,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


# ---------------------------------------------------------------------------
# Candidate.from_dict
# ---------------------------------------------------------------------------


def test_from_dict_basic_fields() -> None:
    c = Candidate.from_dict(_base_dict())
    assert c.candidate_id == "SCHEMA_001"
    assert c.profile.current_title == "Senior ML Engineer"
    assert c.profile.years_of_experience == 6.0
    assert len(c.career_history) == 2
    assert len(c.skills) == 3
    assert c.redrob_signals.notice_period_days == 30


def test_from_dict_salary_parsed() -> None:
    c = Candidate.from_dict(_base_dict())
    assert c.redrob_signals.expected_salary_min == 30.0
    assert c.redrob_signals.expected_salary_max == 55.0


def test_from_dict_sentinels_default() -> None:
    """Missing github and offer fields should default to -1 sentinels."""
    d = _base_dict("SENTINEL_001")
    del d["redrob_signals"]["github_activity_score"]
    del d["redrob_signals"]["offer_acceptance_rate"]
    c = Candidate.from_dict(d)
    assert c.redrob_signals.github_activity_score == -1.0
    assert c.redrob_signals.offer_acceptance_rate == -1.0


def test_from_dict_missing_profile_uses_defaults() -> None:
    """A completely missing profile block should not crash."""
    d = _base_dict("NOPROFILE_001")
    del d["profile"]
    c = Candidate.from_dict(d)
    assert c.profile.current_title == ""


def test_from_dict_missing_redrob_uses_defaults() -> None:
    """A missing redrob_signals block should not crash."""
    d = _base_dict("NOSIGS_001")
    del d["redrob_signals"]
    c = Candidate.from_dict(d)
    assert c.redrob_signals.profile_completeness_score == 0.0


def test_from_dict_empty_collections() -> None:
    d = _base_dict("EMPTY_001")
    d["career_history"] = []
    d["education"] = []
    d["skills"] = []
    d["certifications"] = []
    c = Candidate.from_dict(d)
    assert c.career_history == []
    assert c.skills == []


# ---------------------------------------------------------------------------
# skill_index
# ---------------------------------------------------------------------------


def test_skill_index_merges_assessment_scores() -> None:
    c = Candidate.from_dict(_base_dict())
    idx = skill_index(c)
    # Python exists in skills[] AND assessment_scores -> score should be overlaid
    assert "python" in idx
    assert idx["python"].assessment_score == 88.0
    assert idx["python"].proficiency == "expert"


def test_skill_index_assessment_only_skill() -> None:
    """NumPy is in assessment_scores but NOT in skills[] -> still appears in index."""
    c = Candidate.from_dict(_base_dict())
    idx = skill_index(c)
    assert "numpy" in idx
    assert idx["numpy"].assessment_score == 75.0
    assert idx["numpy"].proficiency == ""  # no proficiency from skills[]


def test_skill_index_no_assessment_score() -> None:
    """Skills without an assessment entry have assessment_score=None."""
    c = Candidate.from_dict(_base_dict())
    idx = skill_index(c)
    assert "faiss" in idx
    assert idx["faiss"].assessment_score is None


# ---------------------------------------------------------------------------
# full_text
# ---------------------------------------------------------------------------


def test_full_text_contains_headline_and_title() -> None:
    c = Candidate.from_dict(_base_dict())
    text = full_text(c)
    assert "AI Engineer at Startup" in text
    assert "Senior ML Engineer" in text


def test_full_text_contains_top_skills_sorted_by_duration() -> None:
    c = Candidate.from_dict(_base_dict())
    text = full_text(c)
    # Skills are emitted as a single chunk sorted by duration DESC.
    # Python (60m) > PyTorch (30m) > FAISS (12m) so the chunk "Python PyTorch FAISS" must appear.
    assert "Python PyTorch FAISS" in text


def test_full_text_contains_education() -> None:
    c = Candidate.from_dict(_base_dict())
    text = full_text(c)
    assert "IIT Bombay" in text


def test_full_text_contains_certification() -> None:
    c = Candidate.from_dict(_base_dict())
    text = full_text(c)
    assert "AWS ML Specialty" in text


def test_full_text_token_cap() -> None:
    """full_text must never exceed the configured token budget."""
    d = _base_dict("LONGTEXT_001")
    # Add a very long summary that would push past 512 words
    d["profile"]["summary"] = " ".join(["word"] * 1000)
    c = Candidate.from_dict(d)
    text = full_text(c)
    assert len(text.split()) <= 512


def test_full_text_no_description_role() -> None:
    """Roles with empty description should still appear (just the title)."""
    d = _base_dict("NODESC_001")
    d["career_history"][0]["description"] = ""
    c = Candidate.from_dict(d)
    text = full_text(c)
    assert "ML Engineer" in text


def test_full_text_empty_candidate() -> None:
    """An empty candidate (all defaults) should not crash."""
    c = Candidate.from_dict({"candidate_id": "EMPTY_TEXT_001"})
    text = full_text(c)
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# scan() -- batch honeypot gate
# ---------------------------------------------------------------------------


def test_scan_returns_only_flagged() -> None:
    clean = Candidate.from_dict(_base_dict("SCAN_CLEAN"))
    flagged_dict = _base_dict("SCAN_FLAGGED")
    flagged_dict["redrob_signals"]["profile_completeness_score"] = 100
    flagged_dict["redrob_signals"]["verified_email"] = False
    flagged_dict["redrob_signals"]["verified_phone"] = False
    flagged = Candidate.from_dict(flagged_dict)

    results = scan([clean, flagged])
    ids = [r.candidate_id for r in results]
    assert "SCAN_FLAGGED" in ids
    assert "SCAN_CLEAN" not in ids


def test_scan_empty_list() -> None:
    assert scan([]) == []
