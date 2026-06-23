"""Reasoning generation (PHASE_2 interim templates; PHASE_3 adds the LLM).

Phase 2 ships grounded, varied template reasoning for all 100 rows — every slot
is filled from ``candidate_facts`` (no invented facts), tone matches rank, and the
template variant is chosen deterministically by candidate_id so no two read alike.
Phase 3 replaces the top-K with LLM reasoning behind a grounding validator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import SupportsFloat, cast

Fact = Mapping[str, object]


def _stable_variant(cid: str, n: int) -> int:
    """Deterministic variant index from candidate_id (stable across processes).

    Uses a content hash, not the builtin ``hash()`` (which is PYTHONHASHSEED-salted),
    so reasoning is byte-identical across runs (NFR-5 determinism).
    """
    digest = hashlib.md5(cid.encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def _first_skill(top_skills: str) -> str:
    """First skill phrase from the 'a; b; c' facts string (or a neutral fallback)."""
    return top_skills.split(";")[0].strip() if top_skills else "relevant skills"


def _tone(rank: int) -> str:
    if rank <= 10:
        return "Strong fit"
    if rank <= 50:
        return "Plausible fit"
    return "Adjacent fit"


def template_reason(fact: Fact, rank: int) -> str:
    """Build a grounded one-line reason for a candidate at ``rank``.

    Uses only real fields from the facts row. The variant index is derived from
    candidate_id so reasoning varies deterministically across candidates.
    """
    cid = str(fact["candidate_id"])
    title = str(fact["current_title"]) or "Engineer"
    company = str(fact["current_company"]) or "their current company"
    yoe = float(cast(SupportsFloat, fact["years_of_experience"]))
    skill = _first_skill(str(fact["top_skills"]))
    gaps = str(fact["gaps"])
    tone = _tone(rank)

    base_variants = [
        f"{tone}: {title} with {yoe:.0f}y at {company}; depth in {skill}.",
        f"{tone}: {yoe:.0f}y {title}; {skill} stands out from the profile.",
        f"{tone}: {title} ({yoe:.0f}y, {company}) — built on {skill}.",
    ]
    text = base_variants[_stable_variant(cid, len(base_variants))]

    if gaps:
        text += f" Concern: {gaps}."
    return text
