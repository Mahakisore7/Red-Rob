# Phase 5 · Sandbox, Polish & Final Submission

**Owner:** All (Mahakisore infra · RamKumar sandbox · Jaswanth docs) · **Duration:** ~Day 4 · **Tag:** `v1.0-final`
**Goal:** Ship the live demo, the submission metadata, the README, and the final validated run.
Make the work legible to judges.

---

## 0. Principle
The model is built; this phase makes it *credible and reproducible*. A live sandbox + clean git
history + honest eval report is what carries Stage 3 and Stage 5.

---

## 1. Streamlit sandbox (`sandbox/streamlit_app.py`)
- Accepts a paste of ≤100 candidate JSON records.
- Runs the **same code path** as `rank.py` (imports `src/aptus/...`) — no divergent logic.
- Returns a ranked table: `rank, candidate_id, title, score, reasoning`.
- Shows the per-candidate signal breakdown (S1–S5 + modifiers) for transparency.
- Deploy to **HuggingFace Spaces** (CPU tier). Confirm cold-start works with the committed/downloaded
  artifacts + GGUF.

```bash
# sandbox/requirements.txt mirrors root pins; Space runs streamlit_app.py
```

---

## 2. `submission_metadata.yaml`
Declare honestly:
```yaml
team: Code Blooded
members: [Mahakisore, Jaswanth Saravanan, RamKumar KR]
approach: dual-retrieval (FAISS+BM25+RRF) + 5-signal composite + local Phi-3-mini rerank
offline_precompute:
  declared: true
  steps: [parse, skill_index, honeypot_gate, text_build, bge-large embed, FAISS, BM25, features, facts, WGT]
  network_used: true   # Phase A only (model download)
models:
  - BAAI/bge-large-en-v1.5 (MIT, embeddings, local at inference)
  - Phi-3-mini-4k-instruct q4_K_M GGUF (MIT, rerank+reasoning, local)
ranking_step:
  network_used: false
  wall_clock_sec: <measured>
  ram_peak_gb: <measured>
artifacts_committed: [bm25.pkl, *_features.parquet, *_facts.parquet, id_map.json, honeypot_ids.json, jd_embedding.npy]
artifacts_rebuilt: [faiss.index]   # too large to commit; precompute.py rebuilds
reproduce: "python precompute.py --candidates data/candidates.jsonl.gz && python rank.py --candidates data/candidates.jsonl.gz --out submission.csv"
llm_blend_weight: <chosen w with holdout NDCG evidence>
```

---

## 3. README (root of the build repo)
- One-paragraph method summary.
- The exact reproduce command (Phase A then Phase B).
- Constraint compliance table (time/RAM/network/disk/determinism) with measured numbers.
- Link to `eval_report.md` (the results table) and the sandbox.
- Hardware notes (offline Phase B, model sources for Phase A).

---

## 4. Final hardening
- [ ] Determinism test green on the final build (`test_determinism.py`).
- [ ] Offline test: run `rank.py` with network disabled — must succeed.
- [ ] Throttled-CPU timing run ≤5 min with adaptive gate.
- [ ] Honeypot assertion (≤3, target 0) in top-100.
- [ ] Manual inspection of top-20: every reasoning specific, grounded, JD-connected, varied.
- [ ] Git history reads as real iteration; each commit has a meaningful message.

---

## 5. Final submission
- Produce `submission.csv` from `v1.0-final`.
- Tag `submission-S3`; record SHA-256 in the commit body.
- **Submit S3** (the last valid submission counts).

---

## 6. Deliverables
- [ ] `sandbox/streamlit_app.py` live on HF Spaces (URL in README + metadata)
- [ ] `submission_metadata.yaml` complete & honest
- [ ] root `README.md` with reproduce command + compliance table
- [ ] final validated `submission.csv` (`submission-S3`)
- [ ] `eval_report.md` final ablation table
- [ ] clean, story-telling git history

---

## 7. Definition of Done (project-level — mirrors PRD §9)
- `rank.py` → 100-row CSV, validator 0 errors, ≤5 min, deterministic, ≤16 GB, offline.
- Internal NDCG@10 ≥ 0.75 on holdout, beating baselines; 0 honeypots in top-100.
- Sandbox live; metadata + README complete; history shows real iteration.

**Git:** merge `phase-5` → `main`, tag `v1.0-final`.

---

## 8. Stage-5 interview prep (the payoff)
Be ready to answer, each backed by an artifact:
| Likely question | Our evidence |
|---|---|
| "Why these weights?" | `jd_requirements.yaml` annotations + `eval_report.md` before/after |
| "How do you beat the sample?" | ablation: naive 0.18 → ours 0.73 NDCG@10 |
| "Why trust the LLM?" | Decision Rule 1: we *measured* it beats composite at w=0.70 |
| "What if the LLM is slow?" | Decision Rule 2: adaptive gate, never below K=10; safety submission exists |
| "Any hallucination risk?" | grounding validator — facts re-checked or we fall back to template |
| "Is it reproducible?" | single command, pinned deps, determinism test, declared metadata |
