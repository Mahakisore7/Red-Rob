"""Aptus-R sandbox (PHASE_5 / docs/02 §2.6) — HuggingFace Spaces demo.

Paste up to ~100 candidate JSON records and the app runs the **same scoring code
path** as ``rank.py`` (embed -> S1-S5 composite -> grounded reasoning) on that small
input, showing a ranked table plus the per-candidate signal breakdown.

This is the small-sample reproducibility check the spec asks for (Section 10.5); it
embeds the pasted candidates fresh, so it needs the ``sandbox``/``precompute`` extras
(torch + sentence-transformers) — it is NOT the timed, offline ``rank.py`` path.

Run locally:  streamlit run sandbox/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``import aptus`` work on a HF Space that just clones the repo (no install).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from aptus import facts, features, honeypot, reasoning, signals  # noqa: E402
from aptus.cli.precompute import _role_scores  # noqa: E402
from aptus.embedder import Embedder  # noqa: E402
from aptus.jd import build_jd_query  # noqa: E402
from aptus.schema import Candidate, full_text  # noqa: E402

_EXAMPLE = (
    '[{"candidate_id": "CAND_0000001", "profile": {"headline": "Senior ML Engineer", '
    '"summary": "Built retrieval and ranking systems.", "location": "Pune", '
    '"country": "India", "years_of_experience": 7, "current_title": "Senior ML Engineer", '
    '"current_company": "Swiggy", "current_company_size": "1001-5000", '
    '"current_industry": "Food Delivery"}, "career_history": [{"company": "Swiggy", '
    '"title": "Senior ML Engineer", "start_date": "2022-01-01", "end_date": null, '
    '"duration_months": 41, "is_current": true, "industry": "Food Delivery", '
    '"company_size": "1001-5000", "description": "Vector search and ranking."}], '
    '"education": [], "skills": [{"name": "Python", "proficiency": "expert", '
    '"endorsements": 30, "duration_months": 84}], "redrob_signals": {'
    '"profile_completeness_score": 90, "signup_date": "2021-01-01", '
    '"last_active_date": "2026-05-20", "open_to_work_flag": true, '
    '"profile_views_received_30d": 40, "applications_submitted_30d": 5, '
    '"recruiter_response_rate": 0.6, "avg_response_time_hours": 8, '
    '"skill_assessment_scores": {"Python": 88}, "connection_count": 400, '
    '"endorsements_received": 30, "notice_period_days": 30, '
    '"expected_salary_range_inr_lpa": {"min": 30, "max": 45}, '
    '"preferred_work_mode": "hybrid", "willing_to_relocate": true, '
    '"github_activity_score": 60, "search_appearance_30d": 80, '
    '"saved_by_recruiters_30d": 12, "interview_completion_rate": 0.9, '
    '"offer_acceptance_rate": 0.5, "verified_email": true, "verified_phone": true, '
    '"linkedin_connected": true}}]'
)


@st.cache_resource  # type: ignore[misc]
def _get_embedder() -> Embedder:
    return Embedder(device="cpu")


def rank_records(records: list[dict]) -> pd.DataFrame:  # type: ignore[type-arg]
    """Score and rank a small list of candidate records (same math as rank.py)."""
    cands = [Candidate.from_dict(r) for r in records]
    emb = _get_embedder()

    cvecs = emb.encode([full_text(c) for c in cands])
    jd_vec = emb.encode([build_jd_query()])[0]
    cosines = cvecs @ jd_vec
    s2_values = _role_scores(cands, emb)
    hp_ids = {r.candidate_id for r in honeypot.scan(cands)}

    rows = []
    for i, c in enumerate(cands):
        s1 = signals.s1_semantic(float(cosines[i]))
        feat = features.build_feature_row(c, s2_values[i], c.candidate_id in hp_ids).as_dict()
        score = signals.final_score(s1, feat)
        fact = facts.build_fact_row(c).as_dict()
        rows.append(
            {
                "candidate_id": c.candidate_id,
                "title": c.profile.current_title,
                "score": round(score, 6),
                "S1": round(s1, 3),
                "S2": round(float(feat["s2_career_arc"]), 3),
                "S3": round(float(feat["s3_behavioral"]), 3),
                "S4": round(float(feat["s4_recency"]), 3),
                "S5": round(float(feat["s5_intent"]), 3),
                "honeypot": bool(feat["is_honeypot"]),
                "_fact": fact,
            }
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["reasoning"] = [
        reasoning.template_reason(r["_fact"], int(rk))
        for rk, r in zip(df["rank"], df.to_dict("records"), strict=False)
    ]
    return df.drop(columns=["_fact"])


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Aptus-R Sandbox", layout="wide")
    st.title("Aptus-R — Candidate Ranking Sandbox")
    st.caption(
        "Paste a JSON array of up to ~100 candidate records. Runs the same 5-signal "
        "scoring as the submission pipeline and shows the per-signal breakdown."
    )
    text = st.text_area("Candidates JSON (array)", value=_EXAMPLE, height=240)

    if st.button("Rank candidates", type="primary"):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return
        if not isinstance(records, list) or not records:
            st.error("Expected a non-empty JSON array of candidate records.")
            return
        if len(records) > 100:
            st.warning(f"{len(records)} records given; scoring the first 100.")
            records = records[:100]

        with st.spinner(f"Embedding + scoring {len(records)} candidates..."):
            df = rank_records(records)
        st.success(f"Ranked {len(df)} candidates.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "score = (0.30 S1 + 0.22 S2 + 0.18 S3 + 0.15 S4 + 0.15 S5) "
            "x modifiers x penalties x honeypot. Honeypots are crushed x0.05."
        )


if __name__ == "__main__":
    main()
