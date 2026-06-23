"""Template reasoning tests (grounded, varied, tone-by-rank)."""

from __future__ import annotations

from aptus import reasoning


def _fact(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "candidate_id": "CAND_0000001",
        "current_title": "Senior ML Engineer",
        "current_company": "Swiggy",
        "years_of_experience": 7.0,
        "top_skills": "Python (expert, 84mo); PyTorch (advanced, 48mo)",
        "gaps": "",
    }
    base.update(overrides)
    return base


def test_reason_is_grounded() -> None:
    # title + first skill appear in every variant; company appears in most.
    text = reasoning.template_reason(_fact(), rank=1)
    assert "Senior ML Engineer" in text
    assert "Python" in text
    assert "7y" in text


def test_tone_matches_rank() -> None:
    assert reasoning.template_reason(_fact(), rank=1).startswith("Strong fit")
    assert reasoning.template_reason(_fact(), rank=30).startswith("Plausible fit")
    assert reasoning.template_reason(_fact(), rank=90).startswith("Adjacent fit")


def test_gaps_surface_as_concern() -> None:
    text = reasoning.template_reason(_fact(gaps="120-day notice"), rank=5)
    assert "Concern: 120-day notice" in text


def test_variation_across_candidates() -> None:
    texts = {
        reasoning.template_reason(_fact(candidate_id=f"CAND_{i:07d}"), rank=20)
        for i in range(1, 30)
    }
    # deterministic templates but should produce more than one distinct phrasing
    assert len(texts) > 1
