"""6-rule integrity gate for Aptus-R v5 (FR-7a through FR-7f).

Flagged candidates are NOT deleted -- they receive a x0.05 score multiplier
at ranking time (FR-7g). This module only decides membership in honeypot_ids.

All thresholds are read from ``config/jd_requirements.yaml`` (FR-12).

Rules
-----
FR-7a  expert + zero duration   : >=2 skills with proficiency=expert AND duration_months=0
FR-7b  career-math mismatch     : sum(duration_months) > yoe*12 + slack_months
FR-7c  too many experts         : count(proficiency=expert) > threshold
FR-7d  perfect-score no-verify  : completeness=100 AND not verified_email AND not verified_phone
FR-7e  keyword stuffer          : non-technical current_title AND >=8 AI/ML skills at adv/expert
FR-7f  assessment contradiction : skill claimed expert but assessment_score < threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aptus.config import HONEYPOT_RULES
from aptus.schema import Candidate, skill_index

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (from YAML -- no magic numbers in code)
# ---------------------------------------------------------------------------

_MIN_EXPERT_ZERO: int = int(HONEYPOT_RULES["fr7a_min_expert_zero_skills"])
_MAX_EXPERT: int = int(HONEYPOT_RULES["fr7c_max_expert_skills"])
_PERFECT_COMPLETENESS: float = float(HONEYPOT_RULES["fr7d_perfect_completeness"])
_MIN_AIML_SKILLS: int = int(HONEYPOT_RULES["fr7e_min_aiml_skills"])
_ADV_EXPERT_LEVELS: set[str] = set(HONEYPOT_RULES["fr7e_proficiency_levels"])
_ASSESSMENT_THRESHOLD: float = float(HONEYPOT_RULES["fr7f_assessment_threshold"])
_SLACK_MONTHS: int = int(HONEYPOT_RULES["fr7b_slack_months"])

# Keywords that indicate a technical title (FR-7e non-technical check)
_TECH_TITLE_KEYWORDS: frozenset[str] = frozenset(
    {
        "engineer",
        "scientist",
        "developer",
        "architect",
        "analyst",
        "researcher",
        "data",
        "ml",
        "ai",
        "nlp",
        "cv",
        "sre",
        "devops",
        "platform",
        "backend",
        "fullstack",
        "frontend",
    }
)

# AI/ML skill name fragments used in FR-7e
_AIML_SKILL_FRAGMENTS: frozenset[str] = frozenset(
    {
        "machine learning",
        "deep learning",
        "neural",
        "pytorch",
        "tensorflow",
        "keras",
        "scikit",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "bert",
        "gpt",
        "llm",
        "transformer",
        "embedding",
        "nlp",
        "natural language",
        "computer vision",
        "cv",
        "reinforcement",
        "rl",
        "generative",
        "diffusion",
        "llama",
        "mistral",
        "hugging",
        "langchain",
        "openai",
        "vector",
        "faiss",
        "retrieval",
        "ranking",
        "recommendation",
    }
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HoneypotResult:
    """Outcome of running the gate on one candidate."""

    candidate_id: str
    is_honeypot: bool
    rules_fired: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------


def _rule_7a(c: Candidate) -> bool:
    """FR-7a: expert proficiency with zero duration on >=2 skills."""
    count = sum(1 for s in c.skills if s.proficiency == "expert" and s.duration_months == 0)
    return count >= _MIN_EXPERT_ZERO


def _rule_7b(c: Candidate) -> bool:
    """FR-7b: total career duration exceeds stated YoE by more than slack."""
    total_months = sum(e.duration_months for e in c.career_history)
    yoe_months = c.profile.years_of_experience * 12
    return total_months > yoe_months + _SLACK_MONTHS


def _rule_7c(c: Candidate) -> bool:
    """FR-7c: more than max_expert_skills skills claimed at expert level."""
    count = sum(1 for s in c.skills if s.proficiency == "expert")
    return count > _MAX_EXPERT


def _rule_7d(c: Candidate) -> bool:
    """FR-7d: perfect completeness score but neither email nor phone verified."""
    sig = c.redrob_signals
    return (
        sig.profile_completeness_score >= _PERFECT_COMPLETENESS
        and not sig.verified_email
        and not sig.verified_phone
    )


def _is_non_technical_title(title: str) -> bool:
    """Return True if the title contains none of the known technical keywords."""
    lower = title.lower()
    return not any(kw in lower for kw in _TECH_TITLE_KEYWORDS)


def _count_aiml_skills_at_level(c: Candidate, levels: set[str]) -> int:
    """Count skills that (a) match an AI/ML fragment and (b) are at the given levels."""
    count = 0
    for s in c.skills:
        if s.proficiency not in levels:
            continue
        name_lower = s.name.lower()
        if any(frag in name_lower for frag in _AIML_SKILL_FRAGMENTS):
            count += 1
    return count


def _rule_7e(c: Candidate) -> bool:
    """FR-7e: non-technical title AND >=8 AI/ML skills at advanced/expert."""
    if not _is_non_technical_title(c.profile.current_title):
        return False
    aiml_count = _count_aiml_skills_at_level(c, _ADV_EXPERT_LEVELS)
    return aiml_count >= _MIN_AIML_SKILLS


def _rule_7f(c: Candidate) -> bool:
    """FR-7f: any skill claimed expert but assessment score < threshold."""
    idx = skill_index(c)
    for ev in idx.values():
        if (
            ev.proficiency == "expert"
            and ev.assessment_score is not None
            and ev.assessment_score < _ASSESSMENT_THRESHOLD
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, object]] = [
    ("FR-7a", _rule_7a),
    ("FR-7b", _rule_7b),
    ("FR-7c", _rule_7c),
    ("FR-7d", _rule_7d),
    ("FR-7e", _rule_7e),
    ("FR-7f", _rule_7f),
]


def check(c: Candidate) -> HoneypotResult:
    """Run all 6 rules against one candidate and return the verdict."""
    fired: list[str] = []
    for rule_id, rule_fn in _RULES:
        try:
            if rule_fn(c):  # type: ignore[operator]
                fired.append(rule_id)
        except Exception as exc:
            logger.warning("candidate %s: rule %s error -- %s", c.candidate_id, rule_id, exc)

    return HoneypotResult(
        candidate_id=c.candidate_id,
        is_honeypot=bool(fired),
        rules_fired=fired,
    )


def scan(candidates: list[Candidate]) -> list[HoneypotResult]:
    """Run the gate over a list of candidates and return results for flagged ones."""
    results: list[HoneypotResult] = []
    for c in candidates:
        result = check(c)
        if result.is_honeypot:
            logger.debug(
                "honeypot flagged %s — rules: %s",
                c.candidate_id,
                ", ".join(result.rules_fired),
            )
            results.append(result)
    return results
