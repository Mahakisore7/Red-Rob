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


# --- grounding validator --------------------------------------------------
def test_is_grounded_accepts_anchored_text() -> None:
    assert reasoning.is_grounded("Strong ML Engineer with Python depth, 7 years.", _fact())


def test_is_grounded_rejects_unanchored() -> None:
    # mentions nothing from the profile (no title word, skill, or company)
    assert not reasoning.is_grounded("A wonderful candidate overall.", _fact())


def test_is_grounded_rejects_wrong_years() -> None:
    # title anchored but claims 15 years vs real 7 -> ungrounded
    assert not reasoning.is_grounded("Senior ML Engineer with 15 years.", _fact())


def test_choose_reasoning_prefers_grounded_llm() -> None:
    llm = "Senior ML Engineer; strong Python and retrieval work."
    assert reasoning.choose_reasoning(_fact(), 1, llm) == llm


def test_choose_reasoning_falls_back_to_template() -> None:
    out = reasoning.choose_reasoning(_fact(), 1, "Totally invented unrelated text.")
    assert out.startswith("Strong fit")  # template tone, not the LLM text
