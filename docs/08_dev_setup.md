# 08 · Developer Environment Setup (zero → running)

**Project:** Aptus-R (v5) · **Team:** Code Blooded
Exact, copy-pasteable steps to go from a clean machine to a reproducible build, using **uv**.
Works on Windows (PowerShell), macOS, and Linux. Where they differ, both are shown.

---

## 1. Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| `uv` | ≥ 0.4 | env + deps + lockfile + run |
| `git` | ≥ 2.40 | version control |
| `make` | any | task shortcuts (optional; Windows via Git Bash or `choco install make`) |
| Docker | ≥ 24 | reproducible offline build (optional but recommended) |

### Install uv
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv --version
```
uv manages Python itself — no system Python needed.

---

## 2. Clone & pin Python

```bash
git clone git@github.com:code-blooded/aptus-r.git
cd aptus-r
echo "3.11" > .python-version
uv python install 3.11        # uv downloads CPython 3.11 if absent
```

---

## 3. `pyproject.toml` (authoritative project definition)

```toml
[project]
name = "aptus-r"
version = "0.1.0"
description = "Intelligent candidate ranking for Redrob Track 1 (Team Code Blooded)"
requires-python = ">=3.11,<3.12"
authors = [
  {name = "Mahakisore"}, {name = "Jaswanth Saravanan"}, {name = "RamKumar KR"},
]
dependencies = [
  "numpy==1.26.4",
  "pandas==2.2.2",
  "pyarrow==16.1.0",
  "faiss-cpu==1.8.0",
  "rank-bm25==0.2.2",
  "llama-cpp-python==0.2.79",
  "pyyaml==6.0.1",
  "orjson==3.10.6",
  "scikit-learn==1.5.1",
  "structlog==24.2.0",
]

[project.optional-dependencies]
# Phase-A-only / heavy deps not needed in the timed Phase B ranking step
precompute = [
  "sentence-transformers==3.0.1",
  "torch==2.3.1",
  "transformers==4.42.4",
]
sandbox = ["streamlit==1.36.0"]

[dependency-groups]
dev = [
  "ruff==0.5.5",
  "mypy==1.11.1",
  "pytest==8.2.2",
  "pytest-cov==5.0.0",
  "pre-commit==3.7.1",
  "nbstripout==0.7.1",
]

[project.scripts]
aptus-precompute = "aptus.cli.precompute:main"
aptus-rank       = "aptus.cli.rank:main"
aptus-eval       = "aptus.cli.eval:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# tool.ruff / tool.mypy / tool.pytest config — see 07_engineering_standards.md
```

> **Design point:** the embedder stack (`torch`, `sentence-transformers`) is an *optional* extra
> (`precompute`). The timed `aptus-rank` path never imports it — it reads precomputed embeddings.
> This keeps the Phase B install small and guarantees no accidental network/model load.

---

## 4. Create the environment & install

```bash
# Full dev environment (everything)
uv sync --all-extras --dev

# OR: only what the timed ranking step needs (lean, offline-safe)
uv sync --frozen                       # runtime deps from uv.lock, no dev

# Phase A machine (needs the embedder extra)
uv sync --extra precompute
```

`uv sync` creates `.venv/`, resolves from `uv.lock` (or writes it first time), and is idempotent.

Verify:
```bash
uv run python -c "import faiss, rank_bm25, llama_cpp, structlog; print('runtime ok')"
uv run python -c "import aptus; print('package importable')"
```

---

## 5. One-time developer setup

```bash
uv run pre-commit install            # install git hooks (lint/format/type gate)
uv run pre-commit run --all-files    # sanity check the whole tree
```

---

## 6. Fetch models (Phase A, network OK)

```bash
# bge-large (embeddings) — cached under models/ via sentence-transformers
uv run --extra precompute python -m aptus.cli.fetch_models

# Phi-3-mini GGUF (rerank) — pinned URL/sha in config; downloaded once
#   stored at models/phi-3-mini-q4.gguf (gitignored)
```
`fetch_models` records each model's SHA-256 in `models/MODEL_MANIFEST.json` for provenance.

---

## 7. Run the pipeline

```bash
# Phase A — once, untimed, network OK
uv run aptus-precompute --candidates data/candidates.jsonl.gz

# Phase B — the timed submission step (≤5 min, CPU, offline)
uv run aptus-rank --candidates data/candidates.jsonl.gz --out submission.csv

# Phase C — internal eval
uv run aptus-eval --submission submission.csv --gold eval/gold_set.csv
```

---

## 8. Makefile shortcuts (wraps uv — the daily interface)

```makefile
.PHONY: setup fmt lint type test cov precompute rank eval check repro

setup:       ; uv sync --all-extras --dev && uv run pre-commit install
fmt:         ; uv run ruff format .
lint:        ; uv run ruff check --fix .
type:        ; uv run mypy src/aptus
test:        ; uv run pytest
cov:         ; uv run pytest --cov-report=html
check:       lint type test          ## the gate CI also runs
precompute:  ; uv run aptus-precompute --candidates data/candidates.jsonl.gz
rank:        ; uv run aptus-rank --candidates data/candidates.jsonl.gz --out submission.csv
eval:        ; uv run aptus-eval --submission submission.csv --gold eval/gold_set.csv
repro:       precompute rank eval     ## full pipeline end-to-end
```

Daily loop: `make fmt lint type test` before every commit (also enforced by pre-commit + CI).

---

## 9. Export `requirements.txt` (for the hackathon, from the lock)

```bash
uv export --no-dev --no-emit-project --format requirements-txt > requirements.txt
```
The lockfile stays authoritative; `requirements.txt` is a generated convenience artifact.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `faiss` import error on Windows | wheel mismatch | use `faiss-cpu==1.8.0` wheel; ensure 3.11 |
| `llama_cpp` build slow | compiling from source | uv pulls prebuilt wheel; ensure recent pip backend |
| lock out of sync | edited pyproject by hand | `uv lock` then commit |
| model download in Phase B | accidental import of `sentence_transformers` | keep it in the `precompute` extra only |
| nondeterministic CSV | threads/seed | confirm `n_threads=1`, `seed=42`, `temperature=0` |
