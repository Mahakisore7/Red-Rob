# 03 · Technology Stack

**Project:** Aptus-R (v5) · **Team:** Code Blooded
Every dependency is justified against a constraint or requirement. Dependencies are declared in
`pyproject.toml` and locked in `uv.lock` (authoritative). See [07_engineering_standards.md](./07_engineering_standards.md)
for the full toolchain and [08_dev_setup.md](./08_dev_setup.md) for setup.

---

## 1. Language & runtime

| Component | Choice | Why |
|---|---|---|
| Language | **Python 3.11** | Mature ML ecosystem; stable wheels. Pinned via `.python-version` + `requires-python` for repro parity |
| Env + deps + lock | **uv** (Astral) | One fast tool for venv, resolution, locking, and `uv run`; `uv.lock` → byte-reproducible deps (NFR-6) |
| Build backend | `hatchling` | standard PEP 517 backend; installs `src/aptus` as a real package |
| Lint + format | **ruff** | replaces flake8/isort/black; one tool, CI-enforced |
| Types | **mypy --strict** | full static typing on `src/aptus` |
| Tests | **pytest + pytest-cov** | ≥85% coverage gate |
| Local gate | **pre-commit** | lint/format/type before every commit |
| Determinism | `PYTHONHASHSEED=0`, fixed seeds, 1-thread LLM | NFR-5 |

> We do **not** use raw `pip`. `requirements.txt` is *exported from the lock* only as a hackathon
> convenience (`uv export`); `uv.lock` remains the source of truth.

---

## 2. Core libraries (declared in `pyproject.toml`, locked in `uv.lock`)

```toml
# [project.dependencies] — needed by the timed Phase B ranking step
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0           # parquet read/write for features & facts
faiss-cpu==1.8.0          # IndexFlatIP, exact cosine on 100K×1024
rank-bm25==0.2.2          # BM25Okapi keyword retrieval
llama-cpp-python==0.2.79  # GGUF inference on CPU, no network
pyyaml==6.0.1
orjson==3.10.6            # fast jsonl parsing
scikit-learn==1.5.1       # ndcg_score, metrics helpers
structlog==24.2.0         # structured logging

# [project.optional-dependencies].precompute — Phase A only (heavy, NOT in rank path)
sentence-transformers==3.0.1
torch==2.3.1              # CPU build; no CUDA required
transformers==4.42.4

# [project.optional-dependencies].sandbox
streamlit==1.36.0         # Phase 5 demo, not in ranking path

# [dependency-groups].dev
ruff==0.5.5  mypy==1.11.1  pytest==8.2.2  pytest-cov==5.0.0  pre-commit==3.7.1  nbstripout==0.7.1
```

> **Design point:** the embedder stack (`torch`, `sentence-transformers`) lives in the **`precompute`
> optional extra**, so the timed `aptus-rank` path (`uv sync --frozen`, no extras) cannot even import
> it. Phase B uses only precomputed embeddings + `faiss-cpu` + `rank-bm25` + `llama-cpp-python` +
> stdlib → lean, offline, deterministic.

---

## 3. Models

| Model | Role | Size | License | Phase | Why this one |
|---|---|---|---|---|---|
| **BAAI/bge-large-en-v1.5** | embeddings (S1, S2) | ~1.3 GB / 1024-d | MIT | A (and JD embed in B uses cached local) | Top-tier retrieval quality; 1024-d separates plain-language experts; precompute is untimed so cost is free |
| **Phi-3-mini-4k-instruct (q4_K_M GGUF)** | top-K rerank + reasoning | ~2.2 GB | MIT | B | Best small instruct model that runs ~3s/call on CPU; strong JSON adherence; MIT license clean |

**Fallback model (declared, optional):** `mistral-7b-instruct q4` if Phi-3 JSON adherence proves
weak on the dataset — decided by measurement in Phase 3/4.

Both models are downloaded **once** in Phase A and cached under `models/`. Phase B loads from disk
with no network call.

---

## 4. Why these vs the alternatives

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Embedder | bge-large (1024-d) | all-MiniLM (384-d, v2-Max) | Better recall; precompute untimed so size is free (keeps v4's choice) |
| Retrieval | FAISS IndexFlatIP + BM25 | pure embedding (v2 option B) | Dual retrieval catches both buzzword users and plain-language experts |
| Fusion | Reciprocal Rank Fusion | weighted score sum | Rank-based fusion is scale-free; no need to normalize disparate score spaces |
| Ranker | hand-weighted composite + earned LLM | LightGBM LambdaMART (v2-High) | ~180 labels would overfit a learned ranker; composite + WGT validation gets ~90% with no training dep |
| Reasoning | local Phi-3 (top-30) + template (31–100) | hosted GPT-4 API | No network in Phase B; API-first teams fail Stage 3 |
| Config | YAML single-source (v2-Max) | weights in code (v4) | One-line tuning, git-diffable, Stage-5 defensible |
| Honeypot | rule gate, flag-not-delete (v4) | hard delete (v1/v2) | False positives stay visible for review |

---

## 5. Hardware envelope

| Resource | Phase A (dev) | Phase B (judge) |
|---|---|---|
| CPU | any modern x86 | unknown; assume constrained — adaptive LLM gate |
| GPU | optional (8 min embed) vs CPU (45–90 min) | **none assumed** |
| RAM | ~8 GB working | ≤ 16 GB hard cap |
| Disk | model+artifact build space | FAISS ~3 GB + GGUF ~2 GB + parquet ~0.1 GB ≈ 5 GB |
| Network | yes (model download) | **none** |

---

## 6. Resource accounting (Phase B, the constrained one)

| Item | RAM | Disk | Load time |
|---|---|---|---|
| FAISS index (memmap) | ~0.4 GB resident | 3 GB | ~12 s |
| BM25 pickle | ~0.3 GB | 0.2 GB | ~4 s |
| features + facts parquet | ~0.2 GB | 0.1 GB | ~2 s |
| Phi-3-mini GGUF (mmap) | ~2.5 GB | 2.2 GB | ~3 s |
| streaming parse buffers | < 1 GB | — | — |
| **Total** | **~4–5 GB** (≪16) | **~5.5 GB** | **~21 s** |

Margins: >3× on RAM, comfortable on disk, ~10× on time before LLM.

---

## 7. Tooling

| Need | Tool |
|---|---|
| Env + deps + lock | **uv** (see [08_dev_setup.md](./08_dev_setup.md)) |
| Lint + format | ruff |
| Types | mypy --strict |
| Tests + coverage | pytest + pytest-cov |
| Local gate | pre-commit |
| CI | GitHub Actions (see [09_cicd_quality.md](./09_cicd_quality.md)) |
| Container | Docker multi-stage (see [10_containerization_repro.md](./10_containerization_repro.md)) |
| Diagrams | Mermaid (in-md) + optional PNG via `@mermaid-js/mermaid-cli` |
| PDF decks | `reportlab` (see `build_pdf.py`) |
| Notebooks (EDA) | Jupyter (nbstripout for clean diffs) |
| Sandbox hosting | HuggingFace Spaces (Streamlit) |
| Version control | git + GitHub; tags per phase (see [06_git_strategy.md](./06_git_strategy.md)) |

---

## 8. Install & sanity (uv)

```bash
# install uv (once) — see 08_dev_setup.md for OS-specific commands
uv python install 3.11

# lean runtime env (what the timed ranking step uses)
uv sync --frozen
uv run python -c "import faiss, rank_bm25, llama_cpp, structlog; print('runtime ok')"

# full dev env (adds embedder extra + sandbox + dev tools)
uv sync --all-extras --dev
uv run pre-commit install
```

Full step-by-step (prereqs, `pyproject.toml`, Makefile, model fetch, troubleshooting) is in
[08_dev_setup.md](./08_dev_setup.md).
