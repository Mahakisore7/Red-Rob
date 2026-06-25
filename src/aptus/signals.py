"""5-signal composite scoring (docs/04 §3-§8, PHASE_2 B6).

Pure arithmetic over a precomputed feature row plus the runtime S1 (semantic)
value. All weights/factors come from ``config/jd_requirements.yaml`` (FR-12).

    final = (sum wi*Si) * prod(modifiers) * prod(penalties) * honeypot_mult
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import SupportsFloat, cast

from aptus.config import HONEYPOT_CFG, PENALTIES, S1_CFG, SIGNAL_WEIGHTS

Feature = Mapping[str, object]

_S1_FLOOR = float(S1_CFG["cosine_floor"])
_S1_RANGE = float(S1_CFG["cosine_range"])
_HP_MULT = float(HONEYPOT_CFG["multiplier"])


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _num(value: object) -> float:
    """Coerce a feature value (numpy/py scalar) to float."""
    return float(cast(SupportsFloat, value))


def s1_semantic(cosine: float) -> float:
    """Stretch raw JD cosine into S1 ∈ [0,1] (docs/04 §3 S1)."""
    return _clip01((cosine - _S1_FLOOR) / _S1_RANGE)


def composite(s1: float, feat: Feature) -> float:
    """Weighted sum of the five signals (S1 runtime, S2-S5 precomputed)."""
    w = SIGNAL_WEIGHTS
    return (
        float(w["s1_semantic"]) * s1
        + float(w["s2_career_arc"]) * _num(feat["s2_career_arc"])
        + float(w["s3_behavioral"]) * _num(feat["s3_behavioral"])
        + float(w["s4_recency"]) * _num(feat["s4_recency"])
        + float(w["s5_intent"]) * _num(feat["s5_intent"])
    )


def modifier_product(feat: Feature) -> float:
    """Product of the four JD modifiers (notice/location/salary/work)."""
    return (
        _num(feat["notice_mod"])
        * _num(feat["location_mod"])
        * _num(feat["salary_mod"])
        * _num(feat["work_mod"])
    )


def penalty_product(feat: Feature) -> float:
    """Product of the disqualifier penalties that apply (docs/04 §6)."""
    factor = 1.0
    if bool(feat["is_consulting_only"]):
        factor *= float(PENALTIES["consulting_only"])
    if bool(feat["is_title_chaser"]):
        factor *= float(PENALTIES["title_chaser"])
    if not bool(feat["has_product_exp"]):
        factor *= float(PENALTIES["no_product_exp"])
    return factor


def honeypot_mult(is_honeypot: bool) -> float:
    """x0.05 for flagged honeypots, else x1.0 (docs/04 §7)."""
    return _HP_MULT if is_honeypot else 1.0


def final_score(s1: float, feat: Feature) -> float:
    """Full master formula: composite * modifiers * penalties * honeypot_mult."""
    base = composite(s1, feat)
    base *= modifier_product(feat)
    base *= penalty_product(feat)
    base *= honeypot_mult(bool(feat["is_honeypot"]))
    return base


def blend(composite_score: float, llm_fit_0_100: float, weight: float) -> float:
    """Blend composite with the LLM fit score (docs/04 §8 top-K blend).

    ``final = w * composite + (1 - w) * (llm_fit / 100)``. ``weight`` is the
    provisional w (Phase 3) finalized by measurement in Phase 4 (DR1).
    """
    return weight * composite_score + (1.0 - weight) * (llm_fit_0_100 / 100.0)
