# 01 · Technical Requirements Document (TRD)

**Project:** Aptus-R (v5) · **Team:** Code Blooded · **Doc version:** 1.0 · **Date:** 2026-06-16
Companion to [00_PRD.md](./00_PRD.md). This doc is the contract the code must satisfy.

---

## 1. System overview

Two-phase (plus an untimed eval phase):

- **Phase A — precompute** (dev machine, untimed, network allowed): parse, gate, embed, index,
  feature-extract, label. Outputs committed artifacts.
- **Phase B — ranking** (`rank.py`, ≤5 min, CPU, no network): load artifacts → retrieve → fuse →
  score → LLM rerank top-K → reason → write CSV → self-validate.
- **Phase C — eval** (`eval.py`, untimed): NDCG/MAP/P@10 + honeypot rate vs gold set.

---

## 2. Functional requirements

### Data & integrity
- **FR-1** Stream-parse `candidates.jsonl.gz` one record at a time; never hold all raw JSON (NFR-2).
- **FR-2** Validate every record against the schema; missing-key records logged, not crashed.
- **FR-3** Build a unified `skill_index()` merging `skills[]` with `redrob_signals.skill_assessment_scores`
  (G3 — closes flaw F5). A skill present only in assessment scores still counts as positive evidence.
- **FR-4** Run a 6-rule integrity gate (FR-7 set below) in Phase A; write `honeypot_ids.json`.

### Honeypot gate (the 6 rules)
- **FR-7a** Expert + zero duration: ≥2 skills with `proficiency=expert` AND `duration_months=0`.
- **FR-7b** Career-math mismatch: `Σ duration_months > years_of_experience×12 + 24`.
- **FR-7c** Too many experts: count(`proficiency=expert`) > 8.
- **FR-7d** Perfect-score-no-verification: `completeness=100` AND `verified_email=False` AND `verified_phone=False`.
- **FR-7e** Keyword stuffer: non-technical title AND ≥8 AI/ML skills at advanced/expert.
- **FR-7f** Assessment contradiction: skill claimed `expert` but `assessment_score < 30`.
- **FR-7g** Handling: do **not** delete. Apply `×0.05` at scoring time; assert top-100 honeypot count ≤3.

### Retrieval
- **FR-5** Embed candidate text + JD with the **same** local model (bge-large-en-v1.5).
- **FR-6** FAISS `IndexFlatIP` on L2-normalized vectors = cosine; return top-500.
- **FR-8** BM25Okapi over tokenized candidate texts; return top-500.
- **FR-9** Reciprocal Rank Fusion: `score = Σ 1/(60+rank_i)`; dedup; keep **top-~500** pool
  (widened from v4's 300, G1 — closes F7). Pool recall validated against WGT positives.

### Scoring
- **FR-10** Compute 5 signals S1–S5 per pooled candidate from precomputed features (arithmetic only).
- **FR-11** Apply modifiers (`notice/location/salary/work`) and disqualifier penalties multiplicatively.
- **FR-12** All weights/thresholds read from `config/jd_requirements.yaml` (G2 — closes F4).
  No magic numbers in code.
- **FR-13** Master formula:
  `final_composite = (Σ wᵢ·Sᵢ) × Π(modifiers) × Π(penalties) × honeypot_mult`.

### LLM rerank & reasoning
- **FR-14** For top-K (K adaptive, default 30): prompt Phi-3-mini (GGUF, local) → JSON
  `{fit_score, hire_recommendation, reasoning}`.
- **FR-15** Decode deterministically: `temperature=0`, fixed `seed`, `n_threads=1` (G4 — closes F8).
- **FR-16** Blend: `final = w·composite + (1−w)·(llm_fit/100)` where `w` is fixed by Decision Rule 1.
- **FR-17** Malformed-JSON fallback: on any parse error, use composite as fit, template reasoning, log.
- **FR-18** Ranks 31–100: template reasoning pulling real candidate fields; no two identical.
- **FR-19** Grounding validator: every slot in every reasoning re-checked against the record;
  fail-loud rather than emit an unverified fact.

### Output
- **FR-20** Write `submission.csv` columns: `candidate_id, rank, score, reasoning`.
- **FR-21** Scores non-increasing by rank; tie-break by `candidate_id` ascending; round to 6 dp.
- **FR-22** Use `csv.writer` (auto-quotes commas in reasoning).
- **FR-23** Auto-run `validate_submission.py`; assert return code 0 before exit.

### Eval
- **FR-24** `eval.py` computes NDCG@10, NDCG@50, MAP, P@10 + honeypot rate vs `gold_set.csv`.
- **FR-25** Eval also runs the naive baseline and a title-only baseline for ablation.

---

## 3. Non-functional requirements

| ID | Requirement | Verification |
|---|---|---|
| **NFR-1** | Ranking wall-clock ≤ 5 min on throttled CPU | timed dry-run; adaptive gate |
| **NFR-2** | Peak RAM ≤ 16 GB | streaming parse, memmap, monitored |
| **NFR-3** | Zero network during Phase B | no HTTP libs imported in `rank.py` path; offline test |
| **NFR-4** | Disk footprint ≤ 5 GB | FAISS ~3 GB + GGUF ~2 GB accounted |
| **NFR-5** | Determinism — byte-identical CSV across runs | `test_determinism.py` |
| **NFR-6** | Single-command reproduction | README + `rank.py` |
| **NFR-7** | All deps pinned | `requirements.txt` with `==` |
| **NFR-8** | Stage-5 explainability | YAML weights + eval report + grounded reasoning |

---

## 4. Interfaces & contracts

### 4.1 CLI
```bash
python precompute.py --candidates PATH [--out-dir artifacts/]
python rank.py       --candidates PATH --out submission.csv [--llm-weight FLOAT] [--top-k INT]
python eval.py       --submission PATH --gold eval/gold_set.csv
```

### 4.2 Artifact contracts (Phase A → Phase B)
| File | Shape / schema | Consumer |
|---|---|---|
| `faiss.index` | 100K × 1024 float32, IndexFlatIP | retriever |
| `bm25.pkl` | BM25Okapi over 100K token lists | retriever |
| `candidate_features.parquet` | 100K rows × {S2, S3, S4, S5 precursors, modifiers, penalty flags} | scorer |
| `candidate_facts.parquet` | 100K rows × {title, company, yoe, top skills, gaps} | reasoning |
| `id_map.json` | FAISS position → candidate_id | retriever |
| `honeypot_ids.json` | list[candidate_id] | scorer (×0.05), assertion |
| `jd_embedding.npy` | 1×1024 float32 | retriever |

**Invariant:** `len(features)==len(facts)==len(id_map)==100000`. Assert at end of Phase A (fail fast).

### 4.3 LLM I/O contract
- **In:** structured prompt (title, company, yoe, location, notice, salary, recent roles, top skills
  sorted by duration desc, github, last_active, open_to_work).
- **Out:** strict JSON `{"fit_score": 0-100, "hire_recommendation": "strong_yes|yes|maybe|no", "reasoning": "<=2 sentences"}`.
- **On violation:** FR-17 fallback.

---

## 5. Data flow (textual; diagrams in [02_architecture.md](./02_architecture.md))

```
candidates.jsonl.gz
  → [A] parse → skill_index → honeypot gate → text_builder
            → embed(bge-large) → FAISS + BM25 + features.parquet + facts.parquet + ids
  → [B] load artifacts → JD embed → FAISS top500 + BM25 top500 → RRF top500
            → 5-signal score × mods × penalties × honeypot_mult
            → top-K → LLM rerank (deterministic) → blend
            → ranks 31–100 template → grounding validator → CSV → validate
  → [C] eval vs gold_set
```

---

## 6. Edge cases (must be handled in code — from v4, retained)

| Case | Fix |
|---|---|
| `offer_acceptance_rate == -1` | neutral 0.5, never negative |
| `github_activity_score == -1` | treat as 0.0, no penalty for no GitHub |
| `end_date == null` & `is_current` | use `duration_months` directly; never compute end−start |
| empty `skills` | skill sub-scores default 0.0; `avg_assessment` defaults 50.0 |
| empty `education` | skip education in text build (not in scoring formula) |
| tie in `final_score` | sort key `(−score, candidate_id)`; round 6 dp |
| comma in reasoning | `csv.writer` quoting |
| malformed LLM JSON | FR-17 fallback |
| candidate in FAISS but not in parquet | assert `len==100000` in Phase A |
| >10 honeypots in top-100 | ×0.05 + terminal assertion ≤3 |

---

## 7. Acceptance tests

| Test | Asserts |
|---|---|
| `test_honeypot.py` | each rule flags its known candidate (CAND_0007353, CAND_0004989, the 21 expert-zero) |
| `test_signals.py` | S1–S5 within [0,1]; modifiers in documented ranges; penalties multiplicative |
| `test_determinism.py` | two `rank.py` runs → byte-identical CSV |
| `test_output.py` | 100 rows, non-increasing scores, valid CSV, validator returns 0 |
| `test_eval.py` | NDCG/MAP math matches a hand-computed fixture |
