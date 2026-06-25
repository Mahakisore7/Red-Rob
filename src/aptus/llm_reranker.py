"""Local Phi-3-mini rerank for the top-K (TRD FR-14/15/16/17, PHASE_3).

Deterministic (temperature=0, fixed seed, single thread), time-bounded (adaptive
K gate, never below min_k), and fail-safe (malformed JSON -> caller falls back to
composite + template). The model call is injected so the logic is testable without
the GGUF; the real generator (``PhiReranker``) is pragma-excluded from coverage.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

from aptus.config import LLM_CFG
from aptus.errors import ModelError

GenerateFn = Callable[[str], str]
Fact = Mapping[str, object]

_VALID_RECS = {"strong_yes", "yes", "maybe", "no"}


@dataclass
class LLMJudgment:
    """One candidate's LLM verdict (or a failed/fallback marker)."""

    candidate_id: str
    fit_score: float | None  # 0-100; None on parse failure
    recommendation: str
    reasoning: str
    ok: bool


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def build_prompt(fact: Fact) -> str:
    """Build the recruiter prompt from real candidate fields (no hallucination input)."""
    return (
        "You are a senior technical recruiter for a Senior AI Engineer role at an AI "
        "startup (Pune/Noida, hybrid, 5-9 years). Ideal: shipped production "
        "retrieval/ranking/recommendation systems, strong Python, vector databases, "
        "embeddings, product-company background.\n\n"
        f"Candidate ID: {fact['candidate_id']}\n"
        f"Title: {fact['current_title']} | Company: {fact['current_company']} | "
        f"Experience: {fact['years_of_experience']} years\n"
        f"Location: {fact['location']} | Relocate: {fact['willing_to_relocate']}\n"
        f"Notice: {fact['notice_period_days']} days\n"
        f"Top skills: {fact['top_skills']}\n"
        f"GitHub: {fact['github_activity_score']} | Last active: {fact['last_active_date']} "
        f"({fact['days_inactive']} days ago)\n"
        f"Known concerns: {fact['gaps'] or 'none'}\n\n"
        'Respond with valid JSON only:\n'
        '{"fit_score": <0-100>, "hire_recommendation": "<strong_yes|yes|maybe|no>", '
        '"reasoning": "<=2 sentences: cite specific facts, connect to the JD, note the '
        'biggest concern>"}'
    )


def parse_response(raw: str) -> dict | None:  # type: ignore[type-arg]
    """Extract the JSON verdict from raw model text; return None on any violation."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "fit_score" not in obj:
        return None
    try:
        fit = float(obj["fit_score"])
    except (TypeError, ValueError):
        return None
    fit = max(0.0, min(100.0, fit))
    rec = str(obj.get("hire_recommendation", "")).lower().strip()
    if rec not in _VALID_RECS:
        rec = ""
    reasoning = str(obj.get("reasoning", "")).strip()
    return {"fit_score": fit, "hire_recommendation": rec, "reasoning": reasoning}


def adaptive_k(
    mean_call_sec: float,
    elapsed_sec: float,
    current_k: int,
    min_k: int,
    budget_sec: float,
    reserve_sec: float,
) -> int:
    """Project how many total LLM calls fit the budget (DR2). Never below min_k."""
    if mean_call_sec <= 0:
        return current_k
    remaining = budget_sec - reserve_sec - elapsed_sec
    affordable = int(remaining / mean_call_sec)
    return max(min_k, min(current_k, affordable))


# ---------------------------------------------------------------------------
# Rerank loop (adaptive, injectable generator)
# ---------------------------------------------------------------------------
def rerank(
    facts: pd.DataFrame,
    candidate_ids: Sequence[str],
    generate: GenerateFn,
    *,
    start_time: float,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, LLMJudgment], int]:
    """LLM-judge candidates in order, shrinking K to stay within the time budget.

    Returns (judgments-by-candidate, chosen_k). ``start_time`` anchors the budget;
    ``clock`` is injectable for deterministic timing tests.
    """
    k = int(LLM_CFG["default_top_k"])
    min_k = int(LLM_CFG["min_top_k"])
    budget = float(LLM_CFG["time_budget_sec"])
    reserve = float(LLM_CFG["assemble_reserve_sec"])

    out: dict[str, LLMJudgment] = {}
    times: list[float] = []
    k_eff = min(k, len(candidate_ids))
    i = 0
    while i < min(k_eff, len(candidate_ids)):
        cid = candidate_ids[i]
        fact = cast("dict[str, object]", facts.loc[cid].to_dict())
        fact["candidate_id"] = cid
        t0 = clock()
        try:
            parsed = parse_response(generate(build_prompt(fact)))
        except Exception:  # - any model error -> fallback (FR-17)
            parsed = None
        times.append(clock() - t0)

        if parsed is not None:
            out[cid] = LLMJudgment(
                cid, parsed["fit_score"], parsed["hire_recommendation"], parsed["reasoning"], True
            )
        else:
            out[cid] = LLMJudgment(cid, None, "", "", False)

        if i == 2:  # after 3 calls, project and (maybe) shrink K
            mean = sum(times) / len(times)
            k_eff = adaptive_k(mean, clock() - start_time, k, min_k, budget, reserve)
        i += 1
    return out, k_eff


# ---------------------------------------------------------------------------
# Real generator (model-backed; not unit-tested)
# ---------------------------------------------------------------------------
class PhiReranker:  # pragma: no cover - requires the local GGUF model
    """Deterministic Phi-3-mini GGUF generator (llama.cpp)."""

    def __init__(self) -> None:
        """Defer the (heavy) model load until the first ``generate`` call."""
        self._llm: object | None = None

    def _load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelError("llama-cpp-python not installed (Phase-3 dep)") from exc
        from pathlib import Path

        from aptus.config import REPO_ROOT

        model_path = REPO_ROOT / str(LLM_CFG["model_path"])
        if not Path(model_path).exists():
            raise ModelError(f"GGUF model not found at {model_path}")
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=int(LLM_CFG["n_ctx"]),
            n_threads=int(LLM_CFG["n_threads"]),
            seed=int(LLM_CFG["seed"]),
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        """Run one deterministic completion and return the raw assistant text."""
        if self._llm is None:
            self._load()
        wrapped = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        out = self._llm(  # type: ignore[operator]
            wrapped,
            max_tokens=int(LLM_CFG["max_tokens"]),
            temperature=float(LLM_CFG["temperature"]),
            top_p=float(LLM_CFG["top_p"]),
            seed=int(LLM_CFG["seed"]),
            stop=["<|end|>"],
        )
        return str(out["choices"][0]["text"])
