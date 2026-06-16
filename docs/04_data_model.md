# 04 · Data Model & Scoring Specification

**Project:** Aptus-R (v5) · **Team:** Code Blooded
This is the precise math contract. `signals.py` must implement exactly this. Reference date for all
"now" calculations: **`REFERENCE_DATE = 2026-06-01`** (`config.py`).

---

## 1. Candidate schema (typed)

From `candidate_schema.json`, wrapped as dataclasses in `schema.py`.

```
Candidate
├─ candidate_id: str
├─ profile: { anonymized_name, headline, summary, location, country,
│             years_of_experience: float, current_title, current_company,
│             current_company_size, current_industry }
├─ career_history: [ CareerEntry{ company, title, start_date, end_date?,
│                    duration_months: int, is_current: bool, industry,
│                    company_size, description } ]
├─ education: [ Education{ institution, degree, field_of_study,
│              start_year, end_year, grade?, tier? } ]
├─ skills: [ Skill{ name, proficiency, endorsements: int, duration_months? } ]
├─ certifications: [ ... ]
├─ languages: [ ... ]
└─ redrob_signals: RedrobSignals{ profile_completeness_score, signup_date,
       last_active_date, open_to_work_flag, profile_views_received_30d,
       applications_submitted_30d, recruiter_response_rate, avg_response_time_hours,
       skill_assessment_scores: {name: 0-100}, connection_count, endorsements_received,
       notice_period_days, expected_salary_min, expected_salary_max, preferred_work_mode,
       willing_to_relocate, github_activity_score, search_appearance_30d,
       saved_by_recruiters_30d, interview_completion_rate, offer_acceptance_rate,
       verified_email, verified_phone, linkedin_connected }
```

### Sentinel handling (DR-1)
| Field | Sentinel | Meaning | Treatment |
|---|---|---|---|
| `github_activity_score` | `-1` | no GitHub linked | use `0.0`, no penalty |
| `offer_acceptance_rate` | `-1` | no offer history | neutral `+0.05` in S5; never negative |

### `skill_index()` — the G3 merge (closes flaw F5)
Union of two evidence sources keyed by skill name:
- from `skills[]`: `proficiency, duration_months, endorsements`
- from `redrob_signals.skill_assessment_scores`: `assessment_score (0–100)`

A skill present **only** in assessment scores still produces a `SkillEvidence` with
`assessment_score` set → counts as positive evidence (the CAND_0006567 case).

---

## 2. Text representation (for embedding, A4)

Concatenate, in order, capped to the 512-token window:
```
headline · summary · current_title · [top 4 career roles: title + 200 chars description]
         · [top 15 skills sorted by duration_months DESC] · [top 2 education] · certifications
```
**Why sorted by duration:** position-weighted embedding models down-weight late tokens, so
`expert, 0 months` honeypot skills (sorted last) contribute weakly while real long-duration skills
lead.

---

## 3. The 5 signals (each ∈ [0,1])

### S1 — Semantic similarity · weight 0.30
FAISS cosine of candidate embedding vs JD embedding, stretched:
```
S1 = clip((cosine − 0.30) / 0.65, 0, 1)
```
Captures conceptual role alignment no keyword list can. Highest weight = core signal.

### S2 — Career arc · weight 0.22  *(hardened vs v4, fixes F6)*
For each past role, embed `title + description` and compare against **multiple JD anchor sentences**
(not one string), then recency-weight:
```
anchors = [ "Senior AI Engineer building retrieval and ranking systems at a product company",
            "engineer shipping embeddings, vector search and recommendation systems",
            "applied ML engineer productionising information retrieval and LLM systems" ]
role_score(r) = mean_over_anchors( cosine(embed(r), embed(anchor)) )
recency_weights = [1.0, 0.8, 0.6, 0.4, 0.2]   # newest→oldest
S2 = Σ recency_weightᵢ · role_scoreᵢ / Σ recency_weightᵢ
```
Concept-thesaurus terms (`concept_thesaurus.yaml`) boost matched roles so the signal isn't hostage
to one wording. Distinguishes *trajectory toward* the role from coincidental keyword overlap.

### S3 — Behavioral signals · weight 0.18
```
S3 = 0.30·min(saved_by_recruiters_30d/20, 1)
   + 0.20·min(search_appearance_30d/100, 1)
   + 0.20·(profile_completeness/100)
   + 0.20·(max(github_activity,0)/100)     # -1 → 0
   + 0.10·min(endorsements_received/50, 1)
```
Market validation: if recruiters are already saving this profile, the crowd agrees.

### S4 — Recency decay · weight 0.15
```
days_inactive = (REFERENCE_DATE − last_active_date).days
S4 = exp(−0.005 × days_inactive)      # 0d→1.0, 90d→0.64, 180d→0.41, 365d→0.16
```
A hireable candidate is an accessible one; year-dormant profiles likely already took an offer.

### S5 — Intent proxy · weight 0.15
```
S5 = 0.30·open_to_work_flag
   + 0.20·min(applications_submitted_30d/10, 1)
   + 0.20·recruiter_response_rate
   + 0.15·interview_completion_rate
   + 0.10·(verified_email AND verified_phone)
   + 0.05·linkedin_connected
   + offer_term            # if offer_acceptance_rate ≥ 0: 0.10·rate ; if -1: +0.05 neutral
```
Are they actually open to hire right now?

---

## 4. Composite

```
composite = 0.30·S1 + 0.22·S2 + 0.18·S3 + 0.15·S4 + 0.15·S5      # ∈ [0,1]
```
All five weights live in `config/jd_requirements.yaml` (G2).

---

## 5. JD modifiers (multiplicative, after composite)

| Modifier | Values |
|---|---|
| `notice_mod` | ≤30d → 1.00 · 31–60 → 0.85 · 61–90 → 0.70 · 91+ → 0.55 |
| `location_mod` | target city (Pune/Noida/NCR/Delhi/Gurgaon/Hyderabad/Mumbai/Bengaluru) → 1.00 · elsewhere+relocate → 0.85 · elsewhere+no-relocate → 0.60 · international → 0.25 |
| `salary_mod` | ≤60 LPA → 1.00 · 61–80 → 0.90 · 81–100 → 0.80 · 100+ → 0.65 |
| `work_mod` | hybrid/flexible → 1.00 · onsite → 0.90 · remote → 0.75 |

---

## 6. Disqualifier penalties (multiplicative, stack)

| Penalty | Factor | Trigger |
|---|---|---|
| `consulting_penalty` | ×0.60 | 100% of career at known service firms (TCS, Infosys, Wipro, Accenture, HCL, …) |
| `title_chaser_penalty` | ×0.75 | avg tenure of last 5 roles < 16 months |
| `no_product_penalty` | ×0.70 | no role at a product company (SaaS/fintech/edtech/consumer-internet/startup) |

---

## 7. Honeypot multiplier

```
honeypot_mult = 0.05 if candidate_id in honeypot_ids else 1.0
```

---

## 8. Final score

```
final_composite = composite × notice_mod × location_mod × salary_mod × work_mod
                  × consulting_penalty × title_chaser_penalty × no_product_penalty
                  × honeypot_mult
```

### Top-K LLM blend (only for the K candidates that reach the reranker)
```
final = w · final_composite + (1 − w) · (llm_fit_score / 100)
```
`w` is **chosen by Decision Rule 1** ([05_eval_framework.md](./05_eval_framework.md)) — default 0.40
only if the LLM beats pure composite on the holdout; else 0.70 (tiebreaker) or 1.0 (LLM dropped).

---

## 9. Ranking & output

- Sort by `final` descending; tie-break `candidate_id` ascending.
- Round score to 6 decimals; enforce non-increasing by rank.
- CSV columns: `candidate_id, rank, score, reasoning`.

---

## 10. Precomputed feature table (`candidate_features.parquet`)

| Column | Type | Source |
|---|---|---|
| `candidate_id` | str | — |
| `s2_career_arc` | float | A8 (embeddings of past roles vs anchors) |
| `s3_behavioral` | float | A8 |
| `s4_recency` | float | A8 |
| `s5_intent` | float | A8 |
| `notice_mod, location_mod, salary_mod, work_mod` | float | A8 |
| `is_consulting_only, is_title_chaser, has_product_exp` | bool | A8 |
| `is_honeypot` | bool | A3 |

> S1 is computed at Phase B (needs the JD query); S2–S5 + modifiers + flags are precomputed so
> Phase B scoring is pure arithmetic.

`candidate_facts.parquet` holds the human-readable fields reasoning needs: title, company, yoe,
top-3 skills (name+prof+months), notice days, location, relocate flag, github, last_active, gaps.
