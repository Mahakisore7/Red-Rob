"""LLM reranker tests: prompt, JSON parsing, adaptive gate, rerank loop (no model)."""

from __future__ import annotations

import pandas as pd

from aptus import llm_reranker as lr


def _facts(n: int = 5) -> pd.DataFrame:
    rows = [
        {
            "candidate_id": f"CAND_{i:07d}",
            "current_title": "Senior ML Engineer",
            "current_company": "Swiggy",
            "years_of_experience": 7.0,
            "location": "Pune",
            "willing_to_relocate": True,
            "notice_period_days": 30,
            "top_skills": "Python (expert, 84mo); FAISS (advanced, 24mo)",
            "github_activity_score": 60.0,
            "last_active_date": "2026-05-20",
            "days_inactive": 12,
            "gaps": "",
        }
        for i in range(1, n + 1)
    ]
    return pd.DataFrame(rows).set_index("candidate_id")


# --- prompt ---------------------------------------------------------------
def test_build_prompt_includes_real_fields() -> None:
    fact = _facts(1).reset_index().to_dict("records")[0]
    p = lr.build_prompt(fact)
    assert "Senior ML Engineer" in p and "Swiggy" in p and "Python" in p
    assert "JSON" in p


# --- parsing --------------------------------------------------------------
def test_parse_valid_json() -> None:
    out = lr.parse_response('{"fit_score": 85, "hire_recommendation": "yes", "reasoning": "ok"}')
    assert out == {"fit_score": 85.0, "hire_recommendation": "yes", "reasoning": "ok"}


def test_parse_json_embedded_in_text() -> None:
    out = lr.parse_response('Sure!\n{"fit_score": 90, "reasoning": "good"} -- done')
    assert out is not None and out["fit_score"] == 90.0
    assert out["hire_recommendation"] == ""  # missing -> normalized empty


def test_parse_clamps_and_rejects() -> None:
    assert lr.parse_response('{"fit_score": 150}')["fit_score"] == 100.0  # clamp
    assert lr.parse_response("not json at all") is None
    assert lr.parse_response('{"reasoning": "no score"}') is None  # missing fit_score


# --- adaptive gate --------------------------------------------------------
def test_adaptive_k_fast_keeps_k() -> None:
    assert lr.adaptive_k(1.0, 3.0, 30, 10, 290, 15) == 30


def test_adaptive_k_slow_shrinks_to_floor() -> None:
    assert lr.adaptive_k(50.0, 100.0, 30, 10, 290, 15) == 10  # never below min_k


def test_adaptive_k_zero_mean_noop() -> None:
    assert lr.adaptive_k(0.0, 0.0, 30, 10, 290, 15) == 30


# --- rerank loop ----------------------------------------------------------
def _grounded_gen(_: str) -> str:
    return '{"fit_score": 80, "hire_recommendation": "yes", "reasoning": "Senior ML Engineer."}'


def test_rerank_fast_clock_judges_all() -> None:
    facts = _facts(5)
    ids = list(facts.index)
    clock = iter(float(i) for i in range(0, 1000))  # tiny call times
    judg, k_eff = lr.rerank(facts, ids, _grounded_gen, start_time=0.0, clock=lambda: next(clock))
    assert all(judg[c].ok and judg[c].fit_score == 80.0 for c in ids)
    assert k_eff >= 5


def test_rerank_malformed_marks_not_ok() -> None:
    facts = _facts(3)
    ids = list(facts.index)
    clock = iter(float(i) for i in range(0, 1000))
    judg, _ = lr.rerank(facts, ids, lambda _: "garbage", start_time=0.0, clock=lambda: next(clock))
    assert all(not judg[c].ok and judg[c].fit_score is None for c in ids)


def test_rerank_slow_clock_shrinks_k() -> None:
    facts = _facts(30)
    ids = list(facts.index)
    # each clock tick jumps 40s -> calls look very slow -> K collapses to the floor
    counter = {"t": 0.0}

    def slow_clock() -> float:
        counter["t"] += 40.0
        return counter["t"]

    _, k_eff = lr.rerank(facts, ids, _grounded_gen, start_time=0.0, clock=slow_clock)
    assert k_eff == int(lr.LLM_CFG["min_top_k"])
