"""Per-candidate feature precompute (PHASE_1 A8, docs/04_data_model.md §3-§7).

Computes everything Phase-B scoring needs as pure arithmetic lookups:
S2-S5, the four JD modifiers, and the three penalty flags. S1 is *not* here —
it needs the JD query and is computed at ranking time.

Every function is pure and unit-tested; all thresholds come from
``config/jd_requirements.yaml`` (FR-12). S2 depends on embeddings, so it is split:
this module owns the role ordering, the role text, the thesaurus boost, and the
recency-weighted aggregation, while ``cli/precompute`` supplies the cosine scores.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import asdict, dataclass

from aptus.config import (
    MODIFIERS,
    PENALTIES,
    REFERENCE_DATE,
    S2_CFG,
    S3_CFG,
    S4_CFG,
    S5_CFG,
    THESAURUS,
)
from aptus.schema import Candidate, CareerEntry

# ---------------------------------------------------------------------------
# Precomputed config-derived constants (read once)
# ---------------------------------------------------------------------------
_SERVICE_FIRMS: tuple[str, ...] = tuple(s.lower() for s in PENALTIES["service_firms"])
_PRODUCT_INDUSTRIES: frozenset[str] = frozenset(s.lower() for s in PENALTIES["product_industries"])
_TARGET_CITIES: tuple[str, ...] = tuple(c.lower() for c in MODIFIERS["location"]["target_cities"])
_RECENCY_WEIGHTS: list[float] = list(S2_CFG["recency_weights"])

# Flatten thesaurus terms -> concept weight (lowercased), for the S2 boost.
_THESAURUS_TERMS: dict[str, float] = {
    term.lower(): float(block["weight"])
    for block in THESAURUS["concepts"].values()
    for term in block["terms"]
}


# ---------------------------------------------------------------------------
# Feature row
# ---------------------------------------------------------------------------
@dataclass
class FeatureRow:
    """One row of ``candidate_features.parquet`` (docs/04 §10)."""

    candidate_id: str
    s2_career_arc: float
    s3_behavioral: float
    s4_recency: float
    s5_intent: float
    notice_mod: float
    location_mod: float
    salary_mod: float
    work_mod: float
    is_consulting_only: bool
    is_title_chaser: bool
    has_product_exp: bool
    is_honeypot: bool

    def as_dict(self) -> dict[str, object]:
        """Return a plain dict (for DataFrame construction)."""
        return asdict(self)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# S2 — career arc (embedding-assisted; aggregation is pure)
# ---------------------------------------------------------------------------
def ordered_roles(c: Candidate) -> list[CareerEntry]:
    """Career roles newest→oldest (current first, then by start_date desc)."""
    return sorted(
        c.career_history,
        key=lambda r: (r.is_current, r.start_date),
        reverse=True,
    )


def role_text(role: CareerEntry) -> str:
    """Text used to embed a single role for S2 (title + description)."""
    return f"{role.title} {role.description}".strip()


def thesaurus_boost(text: str) -> float:
    """Small additive boost if the role text matches JD concept terms.

    Returns the max matched concept weight * 0.05 (so a core-concept match adds
    up to +0.05 before clipping). Keeps S2 from being hostage to one wording.
    """
    lowered = text.lower()
    best = 0.0
    for term, weight in _THESAURUS_TERMS.items():
        if term in lowered and weight > best:
            best = weight
    return 0.05 * best


def aggregate_s2(role_scores: list[float]) -> float:
    """Recency-weighted mean of per-role scores (newest first), clipped to [0,1]."""
    if not role_scores:
        return 0.0
    weights = _RECENCY_WEIGHTS[: len(role_scores)]
    num = sum(w * s for w, s in zip(weights, role_scores, strict=False))
    den = sum(weights)
    return _clip01(num / den) if den else 0.0


# ---------------------------------------------------------------------------
# S3 — behavioral
# ---------------------------------------------------------------------------
def s3_behavioral(c: Candidate) -> float:
    """Market-validation signal (docs/04 §3 S3)."""
    s = c.redrob_signals
    val = (
        float(S3_CFG["saved_by_recruiters_weight"])
        * min(s.saved_by_recruiters_30d / float(S3_CFG["saved_by_recruiters_cap"]), 1.0)
        + float(S3_CFG["search_appearance_weight"])
        * min(s.search_appearance_30d / float(S3_CFG["search_appearance_cap"]), 1.0)
        + float(S3_CFG["completeness_weight"]) * (s.profile_completeness_score / 100.0)
        + float(S3_CFG["github_weight"]) * (max(s.github_activity_score, 0.0) / 100.0)
        + float(S3_CFG["endorsements_weight"])
        * min(s.endorsements_received / float(S3_CFG["endorsements_cap"]), 1.0)
    )
    return _clip01(val)


# ---------------------------------------------------------------------------
# S4 — recency decay
# ---------------------------------------------------------------------------
def days_inactive(c: Candidate, ref: datetime.date = REFERENCE_DATE) -> int:
    """Days since last_active_date; unparseable dates treated as long-dormant."""
    try:
        last = datetime.date.fromisoformat(c.redrob_signals.last_active_date)
    except (ValueError, TypeError):
        return 999
    return max((ref - last).days, 0)


def s4_recency(c: Candidate, ref: datetime.date = REFERENCE_DATE) -> float:
    """Exponential recency decay (docs/04 §3 S4)."""
    lam = float(S4_CFG["decay_lambda"])
    return _clip01(math.exp(-lam * days_inactive(c, ref)))


# ---------------------------------------------------------------------------
# S5 — intent proxy
# ---------------------------------------------------------------------------
def s5_intent(c: Candidate) -> float:
    """Active-intent signal (docs/04 §3 S5). -1 offer rate → neutral half-credit."""
    s = c.redrob_signals
    offer_w = float(S5_CFG["offer_acceptance_weight"])
    # -1 sentinel (no offer history) → neutral half-credit (docs/04 §3 S5).
    rate = s.offer_acceptance_rate if s.offer_acceptance_rate >= 0 else 0.5
    offer_term = offer_w * rate
    val = (
        float(S5_CFG["open_to_work_weight"]) * (1.0 if s.open_to_work_flag else 0.0)
        + float(S5_CFG["applications_weight"])
        * min(s.applications_submitted_30d / float(S5_CFG["applications_cap"]), 1.0)
        + float(S5_CFG["recruiter_response_rate_weight"]) * s.recruiter_response_rate
        + float(S5_CFG["interview_completion_weight"]) * s.interview_completion_rate
        + float(S5_CFG["verified_contact_weight"])
        * (1.0 if (s.verified_email and s.verified_phone) else 0.0)
        + float(S5_CFG["linkedin_weight"]) * (1.0 if s.linkedin_connected else 0.0)
        + offer_term
    )
    return _clip01(val)


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------
def notice_mod(c: Candidate) -> float:
    """Notice-period modifier from notice_period_days."""
    d = c.redrob_signals.notice_period_days
    m = MODIFIERS["notice"]
    if d <= 30:
        return float(m["lte_30"])
    if d <= 60:
        return float(m["lte_60"])
    if d <= 90:
        return float(m["lte_90"])
    return float(m["gt_90"])


def location_mod(c: Candidate) -> float:
    """Location modifier from location/country/willing_to_relocate."""
    m = MODIFIERS["location"]
    country = c.profile.country.strip().lower()
    if country and country != "india":
        return float(m["international"])
    location = c.profile.location.lower()
    if any(city in location for city in _TARGET_CITIES):
        return float(m["target_match"])
    if c.redrob_signals.willing_to_relocate:
        return float(m["elsewhere_relocate"])
    return float(m["elsewhere_no_relocate"])


def salary_mod(c: Candidate) -> float:
    """Salary modifier from the top of the expected range (expected_salary_max)."""
    lpa = c.redrob_signals.expected_salary_max
    m = MODIFIERS["salary_lpa"]
    if lpa <= 60:
        return float(m["lte_60"])
    if lpa <= 80:
        return float(m["lte_80"])
    if lpa <= 100:
        return float(m["lte_100"])
    return float(m["gt_100"])


def work_mod(c: Candidate) -> float:
    """Work-mode modifier from preferred_work_mode."""
    mode = c.redrob_signals.preferred_work_mode.strip().lower()
    m = MODIFIERS["work_mode"]
    return float(m.get(mode, m["onsite"]))


# ---------------------------------------------------------------------------
# Penalty flags
# ---------------------------------------------------------------------------
def _is_service_company(name: str) -> bool:
    n = name.lower()
    return any(firm in n for firm in _SERVICE_FIRMS)


def is_consulting_only(c: Candidate) -> bool:
    """True iff every career role is at a known service firm (≥1 role)."""
    companies = [e.company for e in c.career_history if e.company]
    return bool(companies) and all(_is_service_company(co) for co in companies)


def is_title_chaser(c: Candidate) -> bool:
    """True iff avg tenure of the last N roles is below the threshold.

    Requires ≥3 roles so a single short stint isn't mistaken for job-hopping.
    """
    n_roles = int(PENALTIES["title_chaser_roles"])
    threshold = float(PENALTIES["title_chaser_tenure_months"])
    roles = ordered_roles(c)[:n_roles]
    if len(roles) < 3:
        return False
    avg = sum(r.duration_months for r in roles) / len(roles)
    return avg < threshold


def has_product_exp(c: Candidate) -> bool:
    """True iff any role's (or the current) industry is a product industry."""
    if c.profile.current_industry.strip().lower() in _PRODUCT_INDUSTRIES:
        return True
    return any(e.industry.strip().lower() in _PRODUCT_INDUSTRIES for e in c.career_history)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_feature_row(c: Candidate, s2: float, is_honeypot: bool) -> FeatureRow:
    """Assemble the full feature row (S2 supplied by the embedding pass)."""
    return FeatureRow(
        candidate_id=c.candidate_id,
        s2_career_arc=_clip01(s2),
        s3_behavioral=s3_behavioral(c),
        s4_recency=s4_recency(c),
        s5_intent=s5_intent(c),
        notice_mod=notice_mod(c),
        location_mod=location_mod(c),
        salary_mod=salary_mod(c),
        work_mod=work_mod(c),
        is_consulting_only=is_consulting_only(c),
        is_title_chaser=is_title_chaser(c),
        has_product_exp=has_product_exp(c),
        is_honeypot=is_honeypot,
    )
