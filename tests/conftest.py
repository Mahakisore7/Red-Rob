"""Shared test fixtures: a realistic, valid candidate record factory."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from aptus.schema import Candidate


def _base_record(cid: str = "CAND_0000001") -> dict:  # type: ignore[type-arg]
    """A clean, hireable-looking candidate (product company, available, India)."""
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Candidate A",
            "headline": "Senior ML Engineer building retrieval systems",
            "summary": "Builds ranking and recommendation systems at product companies.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior ML Engineer",
            "current_company": "Swiggy",
            "current_company_size": "1001-5000",
            "current_industry": "Food Delivery",
        },
        "career_history": [
            {
                "company": "Swiggy",
                "title": "Senior ML Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 41,
                "is_current": True,
                "industry": "Food Delivery",
                "company_size": "1001-5000",
                "description": "Built vector search and ranking systems for recommendations.",
            },
            {
                "company": "Flipkart",
                "title": "ML Engineer",
                "start_date": "2018-01-01",
                "end_date": "2021-12-01",
                "duration_months": 47,
                "is_current": False,
                "industry": "E-commerce",
                "company_size": "10001+",
                "description": "Worked on embeddings and recommendation models.",
            },
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2012,
                "end_year": 2016,
                "grade": "8.5",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 84},
            {
                "name": "PyTorch",
                "proficiency": "advanced",
                "endorsements": 20,
                "duration_months": 48,
            },
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2021-01-01",
            "last_active_date": "2026-05-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 40,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.6,
            "avg_response_time_hours": 8.0,
            "skill_assessment_scores": {"Python": 88, "PyTorch": 80},
            "connection_count": 400,
            "endorsements_received": 30,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 60.0,
            "search_appearance_30d": 80,
            "saved_by_recruiters_30d": 12,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


@pytest.fixture
def record_factory() -> Callable[..., dict]:  # type: ignore[type-arg]
    """Return a factory producing fresh base records (override via kwargs in tests)."""
    return _base_record


@pytest.fixture
def candidate_factory() -> Callable[..., Candidate]:
    """Return a factory producing a parsed Candidate from the base record."""

    def _make(cid: str = "CAND_0000001") -> Candidate:
        return Candidate.from_dict(_base_record(cid))

    return _make
