# Phase 4 · Weak Ground Truth, Eval & Tuning

**Owner:** Jaswanth Saravanan · **Duration:** ~Day 3 (parallel) → Day 4 (first half) · **Tag:** `v0.5-tuned`
**Goal:** Build the eval framework (G1), label the WGT, and let measurement — not intuition — decide
every weight and the LLM blend. This phase closes flaws **F1, F2, F7, F10**.

> Labelling starts in parallel during Phase 0–1 (it's the highest-value background task).

---

## 0. Principle
"You cannot optimise a number you never compute." The challenge is 50% NDCG@10. We must measure it
on our own labels before we trust any weight.

---

## 1. Build the Weak Ground Truth (`eval/gold_set.csv`)
Follow [05_eval_framework.md §1](../docs/05_eval_framework.md):
- ~180 candidates, stratified across 5 strata.
- 0–3 rubric, JD-derived.
- 2 labellers on a 60-candidate calibration subset → Cohen's κ → resolve → `labeling_rubric.md`.
- 70/30 split: ~126 tuning, ~54 holdout (read once, at the end).

---

## 2. Build `eval.py`
- NDCG@10, NDCG@50 (`sklearn.metrics.ndcg_score`), MAP, P@10 (hand-rolled, fixture-tested).
- honeypot_rate vs `honeypot_ids.json`.
- Runs 4 rankers for ablation: naive, title-only, composite, composite+LLM.
- Emits a row into `eval/eval_report.md`.
- `tests/test_eval.py` checks the metric math against a tiny hand-computed fixture.

---

## 3. Tuning loop (one weight at a time)
Per [05_eval_framework.md §5](../docs/05_eval_framework.md):
1. Edit one weight in `jd_requirements.yaml`.
2. `rank.py` on the labelled pool → `eval.py`.
3. Keep the change only if NDCG@10 improves on the **tuning** split.
4. Commit with the eval delta in the message; append to `eval_report.md`.

Candidate knobs to sweep: the five S-weights, the modifier breakpoints, the three penalty factors,
the RRF `keep` size, and the honeypot rule thresholds.

---

## 4. Decision Rule 1 — finalize the LLM weight (closes F2)
Evaluate `w ∈ {1.00, 0.70, 0.40}` on the **holdout**:
- Ship the smallest `(1−w)` that maximises holdout NDCG@10.
- Tie / no clear win → `w=1.0` (composite only); LLM stays as reasoning writer.
- Record chosen `w` + numbers in `eval_report.md`; set it in `jd_requirements.yaml`.

## 5. Pool recall check (closes F7)
Assert ≥95% of label-3 candidates appear in the RRF top-500 pool. If not, raise `keep` and re-measure.

## 6. Honeypot calibration (closes F10)
Check the 6 rules against labelled honeypots: 0 false-negatives on label-0 honeypots; minimise
false-positives on label-3s. Adjust thresholds in YAML, re-run.

---

## 7. The Results table (Stage-5 / deck artifact)
Final `eval_report.md` shows the full ablation, e.g.:

| ranker | NDCG@10 | NDCG@50 | MAP | P@10 | honeypot |
|---|---|---|---|---|---|
| naive (sample_submission) | 0.18 | 0.22 | 0.20 | 0.10 | high |
| title-only | 0.55 | 0.58 | 0.50 | 0.50 | 0/100 |
| composite (S1–S5) | 0.70 | 0.68 | 0.63 | 0.70 | 0/100 |
| **composite + LLM (w=0.70)** | **0.73** | **0.69** | **0.64** | **0.80** | **0/100** |

(Numbers illustrative — real values filled during tuning.)

---

## 8. Deliverables
- [ ] `eval/gold_set.csv` (~180 labels) + `labeling_rubric.md` + κ recorded
- [ ] `eval.py` + `tests/test_eval.py` green
- [ ] `eval/eval_report.md` with full tuning history + ablation table
- [ ] `jd_requirements.yaml` weights finalized by measurement
- [ ] chosen LLM weight `w` recorded with holdout evidence
- [ ] pool recall ≥95% on label-3s; honeypot false-negatives = 0

---

## 9. Definition of Done
- NDCG@10 ≥ 0.75 (target) on the holdout, beating naive + title-only baselines.
- LLM weight decided by holdout numbers (not assumed).
- 0 honeypots in top-100; pool recall validated.
- Every shipped weight has a committed before/after in `eval_report.md`.

**Git:** merge `phase-4` → `main`, tag `v0.5-tuned`. Produce `submission-S2` from this tag, record
SHA-256. **Submit S2.**

---

## 10. Risks
| Risk | Mitigation |
|---|---|
| Small labelled set → noisy NDCG | stratify; report holdout separately; prefer lower-variance composite on ties |
| Labeller disagreement | κ + documented resolution; rubric in repo |
| Overfitting weights to tuning split | holdout read once; changes must generalise |
