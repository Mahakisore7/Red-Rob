# Job Description — structured input (T0.5)

**Role:** Senior AI Engineer — Founding Team @ Redrob AI
**Project:** Aptus-R (v5) · **Team:** Code Blooded

> **Provenance.** The organizer ships the raw JD inside the (gitignored) `Dataset/` drop, which is
> not present in this repo. This file consolidates the JD as stated across the challenge brief and
> our specs ([00_PRD.md](../docs/00_PRD.md), [01_TRD.md](../docs/01_TRD.md),
> [04_data_model.md](../docs/04_data_model.md), and the LLM prompt in
> [PHASE_3](../phases/PHASE_3_llm_reasoning.md) §1.2). When the official JD file lands, diff it
> against this and reconcile `jd_requirements.yaml`. This document is the human-readable companion to
> the machine-readable `config/jd_requirements.yaml`, which is the single source of truth at runtime
> (FR-12 / G2).

---

## 1. The role, as structured fields

| Field | Value |
|---|---|
| Title | Senior AI Engineer (Founding Team) |
| Seniority | 5–9 years experience |
| Location | Pune / Noida (NCR); nearby Indian metros acceptable |
| Work mode | **Hybrid** preferred |
| Salary band | **not stated in the JD** — see traceability note in §4.2 |
| Country | India |

## 2. What "ideal" looks like (from the JD / brief)

- Has **shipped production retrieval / ranking / recommendation systems**.
- Strong **Python**; hands-on with **vector databases, embeddings, semantic/vector search**.
- **Product-company** background (not pure services/consulting).
- A career **moving toward** applied ML/IR, not a one-off keyword match.
- **Behaviorally available**: active, open to work, reachable, short notice.

## 3. The six JD skill concepts (→ `concept_thesaurus.yaml`)

These are the canonical concept clusters S2 (career-arc) rewards. Each has a JD-derived weight.

| Concept | Weight | JD rationale |
|---|---|---|
| `retrieval_and_ranking` | 1.00 | Core ask: build retrieval & ranking systems |
| `embeddings_and_vector_db` | 0.90 | "embeddings, vector search, vector databases" |
| `llm_and_nlp` | 0.85 | LLM/NLP systems are part of the founding-team scope |
| `mlops_and_production` | 0.80 | "shipped production … systems" — productionisation |
| `machine_learning_core` | 0.75 | foundational ML competence |
| `data_engineering` | 0.65 | adjacent enabling skill, weakest direct signal |

---

## 4. JD → config traceability (every weight traces to a JD line)

This is the table we defend at Stage 5: each runtime number maps to a JD sentence or doc section.

### 4.1 Signal weights (`signal_weights`)
| Config | Value | Traces to |
|---|---|---|
| `s1_semantic` | 0.30 | "conceptual role alignment no keyword list can capture" — highest weight (meaning over vocabulary) |
| `s2_career_arc` | 0.22 | "career moving toward the role at a product company" — trajectory over title |
| `s3_behavioral` | 0.18 | market validation (recruiters saving, profile views, GitHub) |
| `s4_recency` | 0.15 | "hireable = accessible"; dormant profiles likely already placed |
| `s5_intent` | 0.15 | "actually open to hire right now" |

### 4.2 Modifiers (`modifiers`)
| Config | Traces to JD line |
|---|---|
| `notice.*` | shorter notice = faster to hire for a founding team |
| `location.target_cities` | "Pune / Noida (NCR)"; nearby metros acceptable; international heavily discounted |
| `salary_lpa.*` | ⚠️ **team assumption, NOT a JD line.** The released JD's comp/logistics section states no LPA band. The ≤60→1.0 ladder is a heuristic; revisit (or drop) in Phase 4 tuning. |
| `work_mode.*` | "Hybrid" preferred; onsite ok; remote discounted |

### 4.3 Penalties (`penalties`)
| Config | Value | Traces to JD line |
|---|---|---|
| `consulting_only` | ×0.60 | JD wants **product** ownership, not body-shop/services-only careers |
| `title_chaser` | ×0.75 | senior founding hire needs tenure stability (avg < 16 mo over last 5 roles) |
| `no_product_exp` | ×0.70 | "product-company background" is explicit |

### 4.4 Integrity thresholds (`honeypot_rules`) — defend the adversarial dataset
| Config | Rule | Traces to |
|---|---|---|
| `fr7a_min_expert_zero_skills: 2` | expert + 0 months | impossible profile (honeypot trap class 1) |
| `fr7b_slack_months: 24` | career-math mismatch | Σ duration ≤ YoE×12 + slack |
| `fr7c_max_expert_skills: 8` | too many experts | implausible breadth |
| `fr7d_perfect_completeness: 100` | perfect-score no-verify | gamed completeness |
| `fr7e_min_aiml_skills: 8` | keyword stuffer | non-tech title + ≥8 AI/ML skills (trap class 2) |
| `fr7f_assessment_threshold: 30` | assessment contradiction | claims expert, fails the test |

### 4.5 Output / tie-break (`output`)
| Config | Value | Traces to |
|---|---|---|
| `n_results` | 100 | "output exactly 100 candidates, ranked best-fit first" (hard constraint) |
| `score_decimals` | 6 | determinism / stable comparison (FR-21) |
| `tie_break` | candidate_id_asc | deterministic ordering on equal scores (FR-21) |
| `enforce_non_increasing` | true | ranked output must be monotone non-increasing (FR-21) |

---

## 5. T0.5 completeness checklist (Phase 0 DoD)

- [x] component weights captured (`signal_weights`)
- [x] 6 skill concepts + weights captured (`concept_thesaurus.yaml`)
- [x] all modifiers captured (`modifiers`)
- [x] all penalties captured (`penalties`)
- [x] integrity thresholds captured (`honeypot_rules`)
- [x] tie-break captured (`output`) — **added in T0.5**
- [x] every line annotated with its JD sentence / doc section
- [x] reconciled against the **official organizer JD** (`job_description.docx`, now in the bundle)
- [ ] **follow-up:** `salary_lpa` ladder is not JD-grounded (JD states no band) — tune or drop in Phase 4
- [ ] **follow-up:** JD names disqualifiers not yet modeled (research-only-no-production; "recent
      LangChain/OpenAI only"; CV/speech/robotics without NLP/IR) — candidate Phase-4 penalties
