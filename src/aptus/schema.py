"""Candidate dataclasses and derived helpers for Aptus-R v5.

Wraps ``candidate_schema.json`` fields as typed Python objects (FR-2).
Key public API:
  - ``Candidate.from_dict()``   -- parse one JSON record; missing keys logged, not crashed
  - ``skill_index(c)``          -- G3 merge of skills[] + assessment_scores (FR-3)
  - ``full_text(c)``            -- ordered text blob for embedding (docs/04_data_model.md §2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aptus.config import TEXT_CFG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-record dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CareerEntry:
    """One role in career_history[]."""

    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CareerEntry:
        """Parse a career_history entry; fill defaults for missing keys."""
        return cls(
            company=str(d.get("company", "")),
            title=str(d.get("title", "")),
            start_date=str(d.get("start_date", "")),
            end_date=d.get("end_date"),
            duration_months=int(d.get("duration_months", 0)),
            is_current=bool(d.get("is_current", False)),
            industry=str(d.get("industry", "")),
            company_size=str(d.get("company_size", "")),
            description=str(d.get("description", "")),
        )


@dataclass
class Education:
    """One education record."""

    institution: str
    degree: str
    field_of_study: str
    start_year: int | None
    end_year: int | None
    grade: str | None
    tier: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Education:
        """Parse an education entry."""
        return cls(
            institution=str(d.get("institution", "")),
            degree=str(d.get("degree", "")),
            field_of_study=str(d.get("field_of_study", "")),
            start_year=d.get("start_year"),
            end_year=d.get("end_year"),
            grade=d.get("grade"),
            tier=d.get("tier"),
        )


@dataclass
class Skill:
    """One entry from skills[]."""

    name: str
    proficiency: str  # e.g. "expert", "advanced", "intermediate", "beginner"
    endorsements: int
    duration_months: int  # 0 = not stated

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Skill:
        """Parse a skill entry."""
        return cls(
            name=str(d.get("name", "")),
            proficiency=str(d.get("proficiency", "")).lower(),
            endorsements=int(d.get("endorsements", 0)),
            duration_months=int(d.get("duration_months") or 0),
        )


@dataclass
class RedrobSignals:
    """The redrob_signals block."""

    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]  # name -> 0-100; may be empty
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_min: float  # LPA
    expected_salary_max: float  # LPA
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float  # -1 sentinel = no GitHub
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float  # -1 sentinel = no offer history
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RedrobSignals:
        """Parse redrob_signals; -1 sentinels pass through unchanged (DR-1)."""
        salary = d.get("expected_salary_range_inr_lpa") or {}
        return cls(
            profile_completeness_score=float(d.get("profile_completeness_score", 0)),
            signup_date=str(d.get("signup_date", "")),
            last_active_date=str(d.get("last_active_date", "")),
            open_to_work_flag=bool(d.get("open_to_work_flag", False)),
            profile_views_received_30d=int(d.get("profile_views_received_30d", 0)),
            applications_submitted_30d=int(d.get("applications_submitted_30d", 0)),
            recruiter_response_rate=float(d.get("recruiter_response_rate", 0.0)),
            avg_response_time_hours=float(d.get("avg_response_time_hours", 0.0)),
            skill_assessment_scores=dict(d.get("skill_assessment_scores") or {}),
            connection_count=int(d.get("connection_count", 0)),
            endorsements_received=int(d.get("endorsements_received", 0)),
            notice_period_days=int(d.get("notice_period_days", 90)),
            expected_salary_min=float(salary.get("min", 0)),
            expected_salary_max=float(salary.get("max", 0)),
            preferred_work_mode=str(d.get("preferred_work_mode", "")).lower(),
            willing_to_relocate=bool(d.get("willing_to_relocate", False)),
            github_activity_score=float(d.get("github_activity_score", -1)),
            search_appearance_30d=int(d.get("search_appearance_30d", 0)),
            saved_by_recruiters_30d=int(d.get("saved_by_recruiters_30d", 0)),
            interview_completion_rate=float(d.get("interview_completion_rate", 0.0)),
            offer_acceptance_rate=float(d.get("offer_acceptance_rate", -1)),
            verified_email=bool(d.get("verified_email", False)),
            verified_phone=bool(d.get("verified_phone", False)),
            linkedin_connected=bool(d.get("linkedin_connected", False)),
        )


@dataclass
class Profile:
    """The profile block."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Profile:
        """Parse a profile block."""
        return cls(
            anonymized_name=str(d.get("anonymized_name", "")),
            headline=str(d.get("headline", "")),
            summary=str(d.get("summary", "")),
            location=str(d.get("location", "")),
            country=str(d.get("country", "")),
            years_of_experience=float(d.get("years_of_experience", 0.0)),
            current_title=str(d.get("current_title", "")),
            current_company=str(d.get("current_company", "")),
            current_company_size=str(d.get("current_company_size", "")),
            current_industry=str(d.get("current_industry", "")),
        )


# ---------------------------------------------------------------------------
# Top-level Candidate
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """Full candidate record parsed from candidates.jsonl."""

    candidate_id: str
    profile: Profile
    career_history: list[CareerEntry]
    education: list[Education]
    skills: list[Skill]
    certifications: list[dict[str, Any]]
    languages: list[dict[str, Any]]
    redrob_signals: RedrobSignals

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candidate:
        """Parse one JSON record. Missing top-level keys are logged, not raised."""
        cid = str(d.get("candidate_id", "UNKNOWN"))
        try:
            profile = Profile.from_dict(d.get("profile") or {})
        except Exception as exc:
            logger.warning("candidate %s: bad profile -- %s", cid, exc)
            profile = Profile.from_dict({})

        career: list[CareerEntry] = []
        for entry in d.get("career_history") or []:
            try:
                career.append(CareerEntry.from_dict(entry))
            except Exception as exc:
                logger.debug("candidate %s: skipping career entry -- %s", cid, exc)

        education: list[Education] = []
        for entry in d.get("education") or []:
            try:
                education.append(Education.from_dict(entry))
            except Exception as exc:
                logger.debug("candidate %s: skipping education entry -- %s", cid, exc)

        skills: list[Skill] = []
        for entry in d.get("skills") or []:
            try:
                skills.append(Skill.from_dict(entry))
            except Exception as exc:
                logger.debug("candidate %s: skipping skill entry -- %s", cid, exc)

        try:
            signals = RedrobSignals.from_dict(d.get("redrob_signals") or {})
        except Exception as exc:
            logger.warning("candidate %s: bad redrob_signals -- %s", cid, exc)
            signals = RedrobSignals.from_dict({})

        return cls(
            candidate_id=cid,
            profile=profile,
            career_history=career,
            education=education,
            skills=skills,
            certifications=list(d.get("certifications") or []),
            languages=list(d.get("languages") or []),
            redrob_signals=signals,
        )


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


@dataclass
class SkillEvidence:
    """Merged evidence for one skill (FR-3 / G3 graft -- closes flaw F5)."""

    name: str
    proficiency: str  # from skills[]; empty string if assessment-only
    endorsements: int
    duration_months: int
    assessment_score: float | None  # None if no assessment entry for this skill


def skill_index(c: Candidate) -> dict[str, SkillEvidence]:
    """Build a unified skill map merging skills[] with assessment_scores (FR-3).

    A skill present only in assessment_scores still produces a SkillEvidence
    with assessment_score set, so it counts as positive evidence.
    """
    idx: dict[str, SkillEvidence] = {}

    for s in c.skills:
        key = s.name.lower().strip()
        idx[key] = SkillEvidence(
            name=s.name,
            proficiency=s.proficiency,
            endorsements=s.endorsements,
            duration_months=s.duration_months,
            assessment_score=None,
        )

    for raw_name, score in c.redrob_signals.skill_assessment_scores.items():
        key = raw_name.lower().strip()
        if key in idx:
            idx[key].assessment_score = float(score)
        else:
            idx[key] = SkillEvidence(
                name=raw_name,
                proficiency="",
                endorsements=0,
                duration_months=0,
                assessment_score=float(score),
            )

    return idx


def full_text(c: Candidate) -> str:
    """Build the ordered text blob used for embedding (docs/04_data_model.md §2).

    Order: headline, summary, current_title, top-4 career roles (title + 200-char desc),
    top-15 skills by duration_months DESC, top-2 education, certifications.
    Capped at TEXT_CFG token_budget words.
    """
    max_roles: int = int(TEXT_CFG["max_career_roles"])
    desc_chars: int = int(TEXT_CFG["career_desc_chars"])
    max_skills: int = int(TEXT_CFG["max_skills"])
    max_edu: int = int(TEXT_CFG["max_education"])
    token_budget: int = int(TEXT_CFG["token_budget"])

    parts: list[str] = []

    p = c.profile
    if p.headline:
        parts.append(p.headline)
    if p.summary:
        parts.append(p.summary)
    if p.current_title:
        parts.append(p.current_title)

    for role in c.career_history[:max_roles]:
        role_text = role.title
        if role.description:
            role_text += " " + role.description[:desc_chars]
        parts.append(role_text)

    sorted_skills = sorted(c.skills, key=lambda s: s.duration_months, reverse=True)
    skill_names = [s.name for s in sorted_skills[:max_skills]]
    if skill_names:
        parts.append(" ".join(skill_names))

    for edu in c.education[:max_edu]:
        edu_text = f"{edu.degree} {edu.field_of_study} {edu.institution}".strip()
        if edu_text.strip():
            parts.append(edu_text)

    for cert in c.certifications:
        name = cert.get("name", "")
        if name:
            parts.append(str(name))

    text = " ".join(parts)
    words = text.split()
    if len(words) > token_budget:
        text = " ".join(words[:token_budget])

    return text
