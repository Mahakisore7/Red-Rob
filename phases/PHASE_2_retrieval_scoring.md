# Phase 2 · Retrieval + Scoring + Safety Submission

**Owner:** Mahakisore · **Duration:** ~Day 2 · **Tag:** `v0.3-safety`
**Goal:** A complete **non-LLM** pipeline that produces a valid 100-row CSV. This is the insurance
policy — after this tag, we can never be disqualified on format.

---

## 0. Principle
Build the deterministic core first and commit it as a real submission (`v0.3-safety`). Everything
after (LLM, tuning) is additive and gated by measurement.

---

## 1. `retriever.py` (B2–B5)

```python
def retrieve(jd_text, faiss_index, bm25, id_map, k=500):
    q = embed_local(jd_text)                       # bge-large, cached model
    faiss_ids = faiss_topk(faiss_index, q, k)      # ~0.5s
    bm25_ids  = bm25_topk(bm25, tokenize(jd_text), k)  # ~2s
    return rrf_merge(faiss_ids, bm25_ids, c=60, keep=500)  # pool widened from v4's 300 (fixes F7)
```

**RRF:** `score(cid) = 1/(60+rank_faiss) + 1/(60+rank_bm25)`; dedup; sort desc; keep top-500.
**Recall validation (G1):** once the WGT exists (Phase 4), assert ≥95% of label-3 candidates land in
the pool; raise `keep` if not.

---

## 2. `signals.py` (B6) — the 5-signal score
Implements [04_data_model.md §3–§8](../docs/04_data_model.md) exactly. S1 computed here (needs JD
query); S2–S5 + modifiers + flags read from `candidate_features.parquet` (pure lookup).

```python
def score_candidate(cid, s1, feats, weights):
    composite = (weights.s1*s1 + weights.s2*feats.s2 + weights.s3*feats.s3
               + weights.s4*feats.s4 + weights.s5*feats.s5)
    composite *= feats.notice_mod * feats.location_mod * feats.salary_mod * feats.work_mod
    composite *= consulting_pen(feats) * title_chaser_pen(feats) * no_product_pen(feats)
    composite *= 0.05 if feats.is_honeypot else 1.0
    return composite
```
All weights from `jd_requirements.yaml` (G2). No magic numbers.

---

## 3. `scorer.py` + `output_formatter.py` (B9–B10, non-LLM path)
- Score all ~500 pooled candidates; sort desc; tie-break `candidate_id` asc.
- Take top-100. Round score to 6 dp; enforce non-increasing.
- Reasoning (interim): template-only for all 100 (LLM added in Phase 3).
- Write CSV via `csv.writer`; auto-run `validate_submission.py`; assert return code 0.
- Assert top-100 honeypot count ≤ 3.

---

## 4. `rank.py` wiring (B1–B10, non-LLM)
```bash
python rank.py --candidates data/candidates.jsonl.gz --out submission.csv
```
B1 load → B2–B5 retrieve → B6 score → B9 template reasoning → B10 write + validate.
Time it: should be well under 60 s with no LLM.

---

## 5. Tests
- `tests/test_signals.py` — S1–S5 ∈ [0,1]; modifiers in sets; penalties multiplicative; honeypot ×0.05.
- `tests/test_output.py` — 100 rows, non-increasing scores, validator returns 0, ≤3 honeypots.

---

## 6. Deliverables
- [ ] `retriever.py` (FAISS+BM25+RRF, pool=500)
- [ ] `signals.py` (S1–S5 + modifiers + penalties)
- [ ] `scorer.py`, `output_formatter.py`
- [ ] `rank.py` non-LLM path runnable, < 60 s
- [ ] valid `submission.csv` passing `validate_submission.py`
- [ ] tests green

---

## 7. Definition of Done — **the safety submission**
- `rank.py` produces a 100-row CSV, validator returns 0 errors.
- Top-10 are visibly ML/IR engineers (manual eyeball), not HR Managers.
- ≤3 honeypots in top-100.
- Runs deterministically (same CSV twice — full determinism finalized in Phase 3).

**Git:** merge `phase-2` → `main`, tag `v0.3-safety`. Tag the produced CSV `submission-S1` and record
its SHA-256 in the commit body. **Submit S1.**

---

## 8. Risks
| Risk | Mitigation |
|---|---|
| RRF pool misses a real candidate | widen `keep`; validate recall in Phase 4 |
| Composite ranks a stuffer high | assessment-contradiction rule + role-coherence S2 down-weight |
| Validator format surprise | run it in CI before the tag; copy the exact provided script |
