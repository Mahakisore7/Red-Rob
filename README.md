# Redrob Track 1 — v5 "Aptus-R" — Intelligent Candidate Ranking System

**Team Code Blooded** ·
**Hackathon:** Redrob × Hack2skill "India Runs" — Track 1: The Data & AI Challenge
**Goal:** Rank the 100 best candidates out of 100,000 synthetic profiles for the
*Senior AI Engineer — Founding Team* role at Redrob AI.

---

## What v5 is

v5 is the production-bound design that takes **v4 as the architectural base** (dual retrieval +
RRF + 5-signal composite + local-LLM rerank) and hardens it with **four grafts** drawn from the
earlier explorations:

| Graft | What | Source | Closes |
|---|---|---|---|
| **G1** | Weak Ground Truth eval (NDCG/MAP/P@10 on ~180 hand labels) | v2-High | F1, F7, F10 |
| **G2** | Config-as-YAML (every weight JD-annotated in one file) | v2-Max | F4, helps F6 |
| **G3** | Reuse tested `schema.py` / `honeypot.py` / `config.py` + `skill_index()` | v2-Max | F5, F9 |
| **G4** | Earn the LLM weight (validate blend, adaptive timing, deterministic decode) | v4 + v2-High discipline | F2, F3, F8 |

The full flaw analysis lives in [`v4_flaws_and_v5_integration.pdf`](./v4_flaws_and_v5_integration.pdf).

---

## The hard constraints (never violated)

| Constraint | Value | Where enforced |
|---|---|---|
| Wall-clock (ranking) | ≤ 5 min | Phase B budget table, adaptive LLM gate |
| RAM | ≤ 16 GB | Streaming parse, memmap embeddings |
| Network during ranking | **Zero** | All models local; artifacts precomputed |
| Disk | ≤ 5 GB | bge-large FAISS index ~3 GB + model ~2 GB |
| Output | Exactly 100 rows | `validate_submission.py` auto-run |
| Honeypots in top-100 | ≤ 10 (target 0) | 6-rule gate + ×0.05 penalty + assertion |
| Determinism | same input → same output | fixed seeds, temp=0, single-thread LLM |

---

## Document map

Read in this order.

### Core specs (`docs/`)
| Doc | Purpose |
|---|---|
| [00_PRD.md](./docs/00_PRD.md) | Product Requirements — problem, users, goals, success metrics, scope |
| [01_TRD.md](./docs/01_TRD.md) | Technical Requirements — FR/NFR, interfaces, constraints, acceptance tests |
| [02_architecture.md](./docs/02_architecture.md) | All architecture diagrams (Mermaid) — pipeline, data flow, trap-defeat map |
| [03_tech_stack.md](./docs/03_tech_stack.md) | Full stack (uv toolchain), pinned versions, model choices, rationale |
| [04_data_model.md](./docs/04_data_model.md) | Candidate schema, the 5 signals, modifiers, feature contracts |
| [05_eval_framework.md](./docs/05_eval_framework.md) | Weak Ground Truth, metrics, the two decision rules |
| [06_git_strategy.md](./docs/06_git_strategy.md) | Branching model, commit conventions, version tags, milestone history |
| [07_engineering_standards.md](./docs/07_engineering_standards.md) | Production standards — uv, src-layout, ruff, mypy, logging, errors, testing |
| [08_dev_setup.md](./docs/08_dev_setup.md) | Zero→running with uv: pyproject, Makefile, model fetch, troubleshooting |
| [09_cicd_quality.md](./docs/09_cicd_quality.md) | pre-commit + GitHub Actions + quality gates + determinism/offline CI |
| [10_containerization_repro.md](./docs/10_containerization_repro.md) | Docker multi-stage, offline `--network none`, reproducibility ladder |

### Phase-wise build plan (`phases/`)
| Phase | Doc | Outcome |
|---|---|---|
| 0 | [PHASE_0_foundations.md](./phases/PHASE_0_foundations.md) | Repo, env, data audit, schema/gate reused, EDA |
| 1 | [PHASE_1_precompute.md](./phases/PHASE_1_precompute.md) | All Phase-A artifacts: embeddings, FAISS, BM25, features |
| 2 | [PHASE_2_retrieval_scoring.md](./phases/PHASE_2_retrieval_scoring.md) | Retriever + RRF + 5-signal composite; **safety submission** |
| 3 | [PHASE_3_llm_reasoning.md](./phases/PHASE_3_llm_reasoning.md) | Phi-3-mini rerank, reasoning, adaptive timing, determinism |
| 4 | [PHASE_4_eval_and_tuning.md](./phases/PHASE_4_eval_and_tuning.md) | WGT labelling, eval harness, weight tuning, LLM-weight decision |
| 5 | [PHASE_5_sandbox_polish.md](./phases/PHASE_5_sandbox_polish.md) | Streamlit sandbox, metadata, README, final validated run |

---

## Target repository structure (what we build)

```
aptus-r/
├── pyproject.toml                 # project + deps + ruff/mypy/pytest config + [project.scripts]
├── uv.lock                        # authoritative, committed dependency lock
├── .python-version                # 3.11
├── requirements.txt               # GENERATED from lock (uv export) — hackathon convenience
├── .pre-commit-config.yaml        # local quality gate
├── .github/workflows/             # ci.yml (lint/type/test) + repro.yml (determinism/offline)
├── Makefile                       # task shortcuts wrapping uv
├── Dockerfile  .dockerignore      # multi-stage offline image
├── README.md                      # reproduce command + compliance table
├── submission_metadata.yaml       # declared offline precompute, model list, measured time/RAM
├── validate_submission.py         # provided by Redrob, copied here
│
├── config/
│   ├── jd_requirements.yaml        # G2: every weight, JD-annotated
│   ├── concept_thesaurus.yaml      # 133 skills → 6 JD concepts
│   └── title_taxonomy.yaml         # 47 titles → 4 tiers
│
├── src/aptus/                      # importable package (src-layout)
│   ├── __init__.py
│   ├── config.py                   # G3: loads YAML, constants, paths, seeds
│   ├── schema.py                   # G3: Candidate model, skill_index()
│   ├── honeypot.py                 # G3: 6-rule integrity gate
│   ├── text_builder.py             # candidate → embedding text
│   ├── embedder.py                 # bge-large wrapper (precompute extra only)
│   ├── retriever.py                # FAISS + BM25 + RRF
│   ├── signals.py                  # S1–S5 + modifiers + penalties
│   ├── scorer.py                   # composite assembly
│   ├── llm_reranker.py             # Phi-3-mini, adaptive gate, deterministic
│   ├── reasoning.py                # top-K LLM + rest template + grounding validator
│   ├── output_formatter.py         # CSV writer, tie-break, assertions
│   ├── logging_setup.py            # structured logging (structlog)
│   ├── errors.py                   # typed exception hierarchy
│   └── cli/                        # thin entrypoints → [project.scripts]
│       ├── precompute.py  rank.py  eval.py  fetch_models.py
│
├── eval/
│   ├── gold_set.csv                # G1: ~180 hand labels, 0–3
│   ├── labeling_rubric.md          # how labels were assigned
│   └── eval_report.md              # committed metric history
│
├── tests/                          # mirrors src/aptus; ≥85% coverage gate
│   ├── data/mini.jsonl.gz          # tiny fixture for CI determinism/offline
│   └── test_honeypot.py  test_signals.py  test_determinism.py  test_output.py  test_eval.py
│
├── artifacts/                      # built by aptus-precompute (gitignored)
│   ├── faiss.index  bm25.pkl  candidate_features.parquet
│   ├── candidate_facts.parquet  id_map.json  honeypot_ids.json
│   └── jd_embedding.npy
│
├── models/                         # gitignored; pinned by MODEL_MANIFEST.json (sha256)
│   └── phi-3-mini-q4.gguf          # downloaded once in Phase A
│
├── sandbox/
│   └── streamlit_app.py            # HF Spaces demo (sandbox extra)
│
└── notebooks/
    └── eda.ipynb                   # show-your-work, drives git history
```

---

## Quick reproduce (the single command that matters)

Toolchain is **uv** (not pip). Setup details in [08_dev_setup.md](./docs/08_dev_setup.md).

```bash
# 0) one-time: install uv, then the env (lean runtime; add --extra precompute on the embed box)
uv sync --frozen

# Phase A — once, on dev machine (network OK, untimed)
uv run aptus-precompute --candidates data/candidates.jsonl.gz

# Phase B — the timed submission step (≤5 min, CPU, no network)
uv run aptus-rank --candidates data/candidates.jsonl.gz --out submission.csv

# Phase C — internal eval (untimed)
uv run aptus-eval --submission submission.csv --gold eval/gold_set.csv
```

Or via the Makefile: `make setup` → `make repro`. Fully offline/containerized run:
`docker run --network none ... aptus-r:1.0` (see [10_containerization_repro.md](./docs/10_containerization_repro.md)).

---

## Status legend used across phase docs

- ☐ not started ☑ done ⚠ blocked/at-risk
- Every phase doc ends with a **Definition of Done** and the **git tag** cut at completion.
