# Phase 1 · Offline Precompute (Phase A artifacts)

**Owner:** Mahakisore · **Duration:** ~Day 1 (second half) + overnight embed · **Tag:** `v0.2-precompute`
**Goal:** Produce every artifact `rank.py` will load, so the timed step does no heavy work.

---

## 0. Principle
Everything expensive happens here (untimed, network OK). Phase B becomes "load files + arithmetic +
≤30 LLM calls". This is the constraint-handling backbone (TRD NFR-1/2/3).

---

## 1. `precompute.py` steps (A1–A10)

| Step | What | Output | Est. time |
|---|---|---|---|
| A1 | stream-parse 100K (`orjson`), validate schema, build `id_map` | in memory | ~20 s |
| A2 | `skill_index()` per candidate (merge skills + assessment) | in memory | ~30 s |
| A3 | 6-rule honeypot gate | `honeypot_ids.json` | ~30 s |
| A4 | `text_builder` (headline+summary+roles+skills sorted by duration+edu+certs, ≤512 tok) | in memory | ~2 min |
| A5 | embed 100K with bge-large-en-v1.5, `normalize_embeddings=True` → (100000,1024) f32 | in memory | 45–90 min CPU / ~8 min GPU |
| A6 | FAISS `IndexFlatIP` on normalized vectors; add all; save | `faiss.index` (~3 GB) | ~2 min |
| A7 | tokenize same texts (lowercase, split non-alnum); BM25Okapi; pickle | `bm25.pkl` (~200 MB) | ~5 min |
| A8 | precompute S2–S5 + modifiers + penalty flags + `is_honeypot` | `candidate_features.parquet` (~50 MB) | ~15 min |
| A9 | precompute reasoning facts | `candidate_facts.parquet` (~40 MB) | ~5 min |
| A10 | embed JD anchors + JD doc; save | `jd_embedding.npy` | ~5 s |

### A4 — text builder contract
See [04_data_model.md §2](../docs/04_data_model.md). Cap career history at 4 roles; sort skills by
`duration_months` desc so honeypot `expert, 0mo` skills land last and contribute weakly.

### A5 — embedding
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")   # cached to models/
emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                    show_progress_bar=True).astype("float32")   # (100000, 1024)
```
Run overnight if CPU-only. Cache the model under `models/` for offline Phase B JD embedding.

### A8 — S2 career arc (the hardened signal, fixes F6)
Multi-anchor (3 JD sentences) × recency weights `[1,0.8,0.6,0.4,0.2]`. Thesaurus boost for matched
concepts. Store the scalar `s2_career_arc` per candidate. (Math: [04_data_model.md §3 S2](../docs/04_data_model.md).)

### A8 — modifiers & flags
Compute `notice_mod, location_mod, salary_mod, work_mod` and booleans
`is_consulting_only, is_title_chaser, has_product_exp, is_honeypot`. All thresholds from YAML.

---

## 2. Invariant assertion (closes F-edge "candidate not in parquet")
At end of `precompute.py`:
```python
assert len(features_df) == 100000
assert len(facts_df)   == 100000
assert len(id_map)     == 100000
```
Fail fast in Phase A, never during the timed Phase B.

---

## 3. Sanity checks
- FAISS smoke test: embed the JD, search top-20 — they should be ML/IR engineers, not HR Managers.
- BM25 smoke test: query "vector search embeddings ranking" — top hits keyword-relevant.
- Feature ranges: S2–S5 ∈ [0,1]; modifiers in documented sets.
- `honeypot_ids.json` length ~70–90.

---

## 4. Deliverables
- [ ] `precompute.py` runnable end-to-end with one command
- [ ] `artifacts/`: faiss.index, bm25.pkl, candidate_features.parquet, candidate_facts.parquet,
      id_map.json, honeypot_ids.json, jd_embedding.npy
- [ ] `models/`: bge-large cached, phi-3-mini-q4.gguf downloaded
- [ ] invariant assertions pass; smoke tests documented in notebook

---

## 5. Definition of Done
- `python precompute.py --candidates data/candidates.jsonl.gz` produces all artifacts.
- FAISS top-20 for the JD are visibly relevant ML/IR profiles.
- All length invariants assert true.
- Artifacts either committed (small) or reproducible + declared in `submission_metadata.yaml`
  (FAISS index rebuilt, not committed).

**Git:** merge `phase-1` → `main`, tag `v0.2-precompute`.

---

## 6. Risks
| Risk | Mitigation |
|---|---|
| CPU embed too slow | run overnight; optionally one-time GPU box; cache result |
| 3 GB FAISS too big to commit | rebuild via precompute; document in metadata |
| 512-token truncation drops signal | cap roles at 4, sort skills by duration so best signal leads |
