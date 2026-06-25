# Labeling Rubric — Weak Ground Truth (PHASE_4 / docs/05 §1)

## Status: BOOTSTRAP / auto-generated — needs human correction

`eval/gold_set.csv` was produced by `scripts/build_gold_set.py` using a transparent
rule-based auto-labeler over the Phase-A artifacts (stratified by honeypot + S1-rank
bands; 180 labels). **These are weak labels, not human judgments.**

> ⚠️ **Circularity caveat.** The auto-labeler reuses S1 (semantic similarity), which
> the composite ranker also uses. So the composite scoring well against these labels
> is partly self-fulfilling — that is why `composite` shows NDCG@10 = 1.000 in
> `eval_report.md`. The **naive-vs-composite gap is real** (naive ranks HR Managers;
> composite ranks ML/IR engineers), but the absolute numbers and the LLM-weight
> decision are **not trustworthy until the labels are hand-corrected.**

## The 0-3 scale (JD-derived)

| Score | Meaning |
|---|---|
| **3 — strong fit** | right trajectory + deep skills + product company + available + India + short notice |
| **2 — plausible** | strong technically, one real gap (dormant / consulting-only / international / long notice) |
| **1 — weak/adjacent** | transferable signal, no direct ML/IR production evidence |
| **0 — not a fit** | non-technical, honeypot-shaped, or a core JD disqualifier |

## Auto-labeler heuristic (what the bootstrap applied)

- `0` if honeypot OR non-technical title (tier_5).
- `3` if S1 ≥ 0.60 AND S2 ≥ 0.55 AND product experience AND active AND not consulting-only.
- `2` if (S1 ≥ 0.55 AND S2 ≥ 0.50) OR title tier 1/2.
- `1` if S1 ≥ 0.45 OR title tier 3/4.
- else `0`.

Current distribution: `{0: 59, 1: 12, 2: 47, 3: 62}`.

## How to turn this into a defensible WGT (the human step)

1. Two labellers independently re-score the same ~60-candidate calibration subset by
   reading the actual profiles (not the auto-label).
2. Compute Cohen's κ; discuss disagreements; record resolutions here.
3. Correct the remaining labels (spot-check, fix obvious auto-label errors — especially
   the 2-vs-3 boundary and any tier-5 with real ML history).
4. Hold out 30% (~54) as a test split read only once for the final report.
5. Re-run `scripts/run_ablation.py` — the numbers are now independent and trustworthy.

Until step 5, treat `eval_report.md` as **indicative**: it proves the pipeline runs
end-to-end and that composite >> naive, but not the precise scores or the LLM weight.
