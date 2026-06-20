"""Per-candidate reasoning facts precompute (PHASE_1 A9, docs/04 §10).

Builds the human-readable fields the Phase-3 reasoning layer needs, so reasoning
never has to re-parse raw records. Pure and unit-tested; values are copied
verbatim from the record so the grounding validator can re-check them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from aptus import features
from aptus.schema import Candidate, Skill

#: How many top skills to surface for reasoning.
TOP_SKILLS = 3


@dataclass
class FactRow:
    """One row of ``candidate_facts.parquet`` (reasoning inputs)."""

    candidate_id: str
    current_title: str
    current_company: str
    years_of_experience: float
    top_skills: str
    notice_period_days: int
    location: str
    willing_to_relocate: bool
    github_activity_score: float
    last_active_date: str
    days_inactive: int
    gaps: str

    def as_dict(self) -> dict[str, object]:
        """Return a plain dict (for DataFrame construction)."""
        return asdict(self)


def _format_skill(s: Skill) -> str:
    """Render one skill as ``name (proficiency, Nmo)``."""
    prof = s.proficiency or "unrated"
    return f"{s.name} ({prof}, {s.duration_months}mo)"


def top_skills_str(c: Candidate, k: int = TOP_SKILLS) -> str:
    """Top-k skills by duration_months desc, as a single grounded string."""
    ranked = sorted(c.skills, key=lambda s: s.duration_months, reverse=True)[:k]
    return "; ".join(_format_skill(s) for s in ranked)


def gaps_str(c: Candidate) -> str:
    """Concise, grounded list of hiring concerns (drives honest reasoning)."""
    concerns: list[str] = []
    if c.redrob_signals.notice_period_days > 60:
        concerns.append(f"{c.redrob_signals.notice_period_days}-day notice")
    if features.days_inactive(c) > 180:
        concerns.append(f"dormant {features.days_inactive(c)}d")
    if features.is_consulting_only(c):
        concerns.append("consulting-only career")
    if not features.has_product_exp(c):
        concerns.append("no product-company experience")
    if features.location_mod(c) < 0.7:
        concerns.append("location/relocation friction")
    if features.is_title_chaser(c):
        concerns.append("short average tenure")
    return "; ".join(concerns)


def build_fact_row(c: Candidate) -> FactRow:
    """Assemble the reasoning fact row for one candidate."""
    s = c.redrob_signals
    return FactRow(
        candidate_id=c.candidate_id,
        current_title=c.profile.current_title,
        current_company=c.profile.current_company,
        years_of_experience=c.profile.years_of_experience,
        top_skills=top_skills_str(c),
        notice_period_days=s.notice_period_days,
        location=c.profile.location,
        willing_to_relocate=s.willing_to_relocate,
        github_activity_score=s.github_activity_score,
        last_active_date=s.last_active_date,
        days_inactive=features.days_inactive(c),
        gaps=gaps_str(c),
    )
