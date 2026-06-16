# 00 · Product Requirements Document (PRD)

**Project:** Aptus-R — Intelligent Candidate Ranking System (v5)
**Team:** Code Blooded
**Owner:** Mahakisore (eng lead) · Jaswanth Saravanan (eval/data) · RamKumar KR (LLM/reasoning)
**Status:** Approved for build · **Doc version:** 1.0 · **Date:** 2026-06-16

---

## 1. Problem statement

Redrob is a hiring platform. Their current candidate ranking is BM25 + rule-based scoring — it
works but surfaces the wrong people. The hackathon gives us **100,000 synthetic candidate
profiles** and **one job description** (*Senior AI Engineer — Founding Team*, Pune/Noida, hybrid,
5–9 years). We must output a CSV of **exactly 100 candidates, ranked best-fit first**, each with a
score and a human-readable reasoning string.

The dataset is **adversarial by design**. The provided `sample_submission.csv` ranks an HR Manager
at #1 because it uses a naive formula (AI-skill-count × recruiter-response-rate) with no concept of
role fit. Beating that benchmark is the entire game.

---

## 2. Who the "user" is

| Persona | Need | How v5 serves it |
|---|---|---|
| **Redrob hiring manager** (in-fiction) | A shortlist of genuinely hireable Senior AI Engineers | 5-signal composite that models role fit, trajectory, and availability |
| **Hackathon judge (Stage 3)** | Reproducible pipeline, high NDCG@10/@50 | Single-command run, eval harness, pinned deps |
| **Hackathon judge (Stage 5)** | Defensible design choices in a live interview | YAML-annotated weights, eval numbers, git history |

---

## 3. The adversarial trap classes (what we must defeat)

| # | Trap | Example | v5 defence |
|---|---|---|---|
| 1 | **Honeypots** (~80) | Subtly impossible profiles: expert skill with 0 months, 8 years at a 3-year-old company | 6-rule integrity gate (G3), ×0.05 penalty, final assertion |
| 2 | **Keyword stuffers** | HR Manager listing 10 AI skills at "expert" | role-coherence (S2) + assessment-contradiction rule |
| 3 | **Plain-language Tier-5s** | Real expert who writes "search backend", never "RAG" | semantic retrieval (S1) + concept thesaurus |
| 4 | **Behavioral twins** | Two identical-on-paper candidates, one dormant | recency (S4) + intent (S5) separation |

---

## 4. Goals (G) and non-goals

### Goals
- **G1 — Beat the naive baseline** on NDCG@10/@50 by a measurable margin on our own labelled set.
- **G2 — Zero honeypots** in the top-100 (hard requirement: ≤10, target 0).
- **G3 — Surface real fit:** top-10 are 6–8 yr ML/IR engineers with product-company, shipped
  retrieval/ranking systems, behaviorally available, India-based, short notice.
- **G4 — Survive the constraints:** ≤5 min, ≤16 GB, zero network, deterministic.
- **G5 — Stage-5 defensible:** every weight traces to a JD line; every claim in reasoning is grounded.
- **G6 — Reproducible:** one command, pinned deps, committed artifacts or a download script.

### Non-goals
- Not building a general-purpose ATS. Single JD, single run.
- Not training a large model. Local inference only; no fine-tuning required.
- Not optimising ranks 51–100 beyond template reasoning (NDCG@10 is 50% of score; focus there).

---

## 5. Success metrics

| Metric | Target | Measured by |
|---|---|---|
| NDCG@10 (internal) | ≥ 0.75 on WGT holdout | `eval.py` |
| NDCG@50 (internal) | ≥ 0.70 on WGT holdout | `eval.py` |
| Honeypot rate in top-100 | 0 (max 10) | assertion vs `honeypot_ids.json` |
| Ranking wall-clock | ≤ 5 min on throttled CPU | timed dry-run |
| Reasoning grounding | 100% of facts verifiable | grounding validator |
| Determinism | byte-identical CSV across 2 runs | `test_determinism.py` |

Composite challenge score (organizer formula):
`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`.

---

## 6. The product decision: what makes a candidate "good"

Encoded as the 5-signal composite (full math in [04_data_model.md](./04_data_model.md)):

```
composite = 0.30·S1(semantic) + 0.22·S2(career-arc) + 0.18·S3(behavioral)
          + 0.15·S4(recency)  + 0.15·S5(intent)
final = composite × notice_mod × location_mod × salary_mod × work_mod × disqualifier_penalties
```

Plus, for the top-K only: `final = w·composite + (1−w)·llm_fit`, where **w is chosen by measurement**
(Decision Rule 1, [05_eval_framework.md](./05_eval_framework.md)).

Guiding principles:
1. **Verification before scoring** — honeypots are gated out before they can rank.
2. **Trajectory over title** — a career *moving toward* the role beats a coincidental keyword match.
3. **Meaning over vocabulary** — semantic retrieval catches plain-language experts.
4. **Availability is a multiplier, not a gate** — dormant perfect-fits sink, but are never zeroed.
5. **Measurement decides** — no weight or model ships unless it wins on the labelled set.

---

## 7. Scope & timeline (maps to phase docs)

| Phase | Deliverable | Owner | Doc |
|---|---|---|---|
| 0 | Foundations: repo, env, EDA, reused schema/gate | Mahakisore | PHASE_0 |
| 1 | Precompute artifacts | Mahakisore | PHASE_1 |
| 2 | Retrieval + scoring + **safety submission** | Mahakisore | PHASE_2 |
| 3 | LLM rerank + reasoning | RamKumar | PHASE_3 |
| 4 | WGT + eval + tuning + LLM-weight decision | Jaswanth | PHASE_4 |
| 5 | Sandbox + polish + final run | All | PHASE_5 |

---

## 8. Risks & mitigations (product-level)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM blows time budget on judge HW | Med | High | Adaptive gate (30→20→15→10); Day-2 non-LLM safety submission committed |
| LLM hurts ranking vs composite | Med | High | Decision Rule 1: keep LLM only if it wins on holdout |
| Honeypot false-negatives → DQ | Low | Critical | 6 rules + ×0.05 + terminal assertion + eval-set rate check |
| Over-engineering eats the 4 days | Med | Med | Strict non-goals; safety submission by end of Phase 2 |
| Non-deterministic output rejected | Low | High | temp=0, fixed seed, single-thread; `test_determinism.py` |

---

## 9. Acceptance (Definition of Done for the product)

- [ ] `rank.py` produces a 100-row CSV that passes `validate_submission.py` with 0 errors.
- [ ] Internal NDCG@10 ≥ 0.75 on the WGT holdout, beating the naive baseline.
- [ ] 0 honeypots in top-100 (assertion passes).
- [ ] Full run ≤ 5 min on a throttled CPU; determinism test passes.
- [ ] Streamlit sandbox live on HF Spaces.
- [ ] Git history shows real iteration (EDA → gate → retriever → scorer → reranker → polish).
- [ ] `submission_metadata.yaml` + README complete.
