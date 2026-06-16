# 07 · Engineering Standards (production-grade)

**Project:** Aptus-R (v5) · **Team:** Code Blooded
This is the non-negotiable engineering baseline. Every file we write conforms to it. The goal: a
repo that looks like it came out of a serious ML team, not a hackathon weekend — because Stage 3
grades reproducibility and code authenticity, and Stage 5 grades whether we can defend our choices.

> TL;DR toolchain: **uv** (env + deps + lockfile) · **ruff** (lint + format) · **mypy** (types) ·
> **pytest + coverage** · **pre-commit** (local gate) · **GitHub Actions** (CI gate) ·
> **conventional commits + SemVer tags** · **structured logging** · **typed, documented code**.

---

## 1. Package & environment management — `uv`

We use [`uv`](https://docs.astral.sh/uv/) (Astral) instead of pip/venv/poetry. Rationale:
- One tool for venv creation, dependency resolution, locking, and running.
- 10–100× faster installs → fast CI and fast onboarding.
- A committed `uv.lock` gives **byte-for-byte reproducible** dependency trees (supports NFR-5/6).
- `uv run` executes inside the project env without manual activation — kills "works on my machine".

### Rules
- **Never** call `pip install` directly. Use `uv add` / `uv remove`.
- `pyproject.toml` is the single source of dependency truth; `uv.lock` is committed.
- Pin Python with `.python-version` (3.11) and `requires-python = ">=3.11,<3.12"`.
- Production deps and dev deps are separated (`[project.dependencies]` vs `[dependency-groups].dev`).
- A `requirements.txt` is **exported** from the lock (`uv export`) only because the hackathon may
  expect it; the lockfile remains authoritative.

```bash
uv sync --frozen            # install exactly from uv.lock (CI + judge repro)
uv add faiss-cpu            # add a runtime dep (updates pyproject + lock)
uv add --dev ruff mypy      # add a dev-only dep
uv run rank.py --help       # run inside the env, no activation
uv export --no-dev --format requirements-txt > requirements.txt
```

---

## 2. Project layout — `src/` layout (importable package)

We use the **src layout** so tests run against the *installed* package, not loose files — this
catches packaging bugs early and is the modern Python standard.

```
aptus-r/
├── pyproject.toml            # project metadata, deps, tool config (ruff/mypy/pytest)
├── uv.lock                   # committed, authoritative dependency lock
├── .python-version           # 3.11
├── .pre-commit-config.yaml
├── .gitignore  .dockerignore  .editorconfig
├── Makefile                  # task shortcuts (wraps uv)
├── Dockerfile                # reproducible offline build
├── README.md
├── submission_metadata.yaml
├── config/                   # YAML knobs (jd_requirements, thesaurus, taxonomy)
├── src/aptus/                # the package (importable as `aptus`)
│   ├── __init__.py
│   ├── config.py  schema.py  honeypot.py  text_builder.py
│   ├── embedder.py  retriever.py  signals.py  scorer.py
│   ├── llm_reranker.py  reasoning.py  output_formatter.py
│   ├── logging_setup.py      # structured logging config
│   ├── errors.py             # typed exception hierarchy
│   └── cli/                  # thin CLI entrypoints
│       ├── precompute.py  rank.py  eval.py
├── tests/                    # mirrors src/aptus structure
├── eval/                     # gold_set.csv, rubric, eval_report.md
├── artifacts/  models/       # gitignored (rebuilt/downloaded)
├── sandbox/                  # streamlit app
└── notebooks/                # eda.ipynb
```

CLI entrypoints are registered in `pyproject.toml` so they become real commands:
```toml
[project.scripts]
aptus-precompute = "aptus.cli.precompute:main"
aptus-rank       = "aptus.cli.rank:main"
aptus-eval       = "aptus.cli.eval:main"
```
→ `uv run aptus-rank --candidates ... --out submission.csv`.

---

## 3. Code style & linting — `ruff`

Ruff is both linter and formatter (replaces flake8 + isort + black). Config lives in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "C4", "SIM", "PTH", "RUF", "ANN", "D"]
ignore = ["D203", "D213", "ANN101", "ANN102"]   # docstring/style conflicts
[tool.ruff.lint.pydocstyle]
convention = "google"
[tool.ruff.lint.per-file-ignores]
"tests/*" = ["ANN", "D"]      # tests need not be fully typed/documented
```

- `ruff format` is the only formatter; no manual style debates.
- `ruff check --fix` auto-fixes imports, unused vars, simplifications.
- CI fails on any lint error.

---

## 4. Type safety — `mypy` (strict)

All `src/aptus` code is fully type-annotated and passes `mypy --strict`.

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true
disallow_untyped_defs = true
plugins = []
[[tool.mypy.overrides]]
module = ["faiss.*", "rank_bm25.*", "llama_cpp.*", "sentence_transformers.*"]
ignore_missing_imports = true     # third-party libs without stubs
```

- Public functions declare full signatures and return types.
- Domain types are modelled with `dataclass`/`TypedDict`/`Enum`, not bare dicts.
- No `Any` in business logic except at I/O boundaries (raw JSON parsing), immediately narrowed.

---

## 5. Error handling — typed exception hierarchy

No bare `except:`. A small typed hierarchy in `errors.py`:

```python
class AptusError(Exception): ...
class SchemaValidationError(AptusError): ...      # malformed candidate record
class ArtifactMissingError(AptusError): ...       # expected artifact not found
class IntegrityGateError(AptusError): ...         # gate misconfiguration
class LLMDecodeError(AptusError): ...             # malformed model output (→ fallback)
class BudgetExceededError(AptusError): ...        # timing guard tripped
```

Principles:
- **Fail fast in Phase A** (precompute), **degrade gracefully in Phase B** (ranking).
  e.g. `LLMDecodeError` → composite + template reasoning (never crash the run).
- Every caught exception is logged with context; never silently swallowed.
- Invariants asserted explicitly (`assert len(features)==100000`) with a message.

---

## 6. Logging & observability — structured

Use `structlog` (or stdlib `logging` with a JSON formatter) configured once in `logging_setup.py`.

```python
log = get_logger(__name__)
log.info("retrieval.done", faiss_ms=512, bm25_ms=1980, pool_size=500)
log.warning("llm.json_fallback", candidate_id=cid)
log.info("phaseB.timing", step="B7_llm", k=20, mean_call_ms=3100, budget_left_s=110)
```

- **No `print()`** in `src/` — only structured logs.
- Phase B logs a final timing table (per-step ms) → feeds the metadata + Stage-5 defense.
- Log levels: DEBUG (dev), INFO (default run), WARNING (degradations), ERROR (aborts).
- A `--verbose` flag bumps to DEBUG; default INFO writes a compact run summary.

---

## 7. Determinism (production reproducibility)

Centralised in `config.py` and asserted in CI (`test_determinism.py`):
```python
os.environ["PYTHONHASHSEED"] = "0"
random.seed(SEED); np.random.seed(SEED)          # SEED = 42
# llama.cpp: seed=42, n_threads=1, temperature=0.0
```
- Two `aptus-rank` runs must produce a byte-identical CSV.
- The `uv.lock` + pinned model files make the *whole stack* reproducible, not just the seeds.

---

## 8. Testing standards — `pytest` + coverage

- `tests/` mirrors `src/aptus/`. Naming: `test_<module>.py`.
- **Coverage gate:** ≥85% on `src/aptus` (CI fails below).
- Test taxonomy:
  | Kind | Examples |
  |---|---|
  | unit | each honeypot rule, each signal range, RRF math, tie-break |
  | property | signals always ∈ [0,1]; scores non-increasing after sort |
  | golden/fixture | NDCG/MAP vs hand-computed fixture; known honeypot IDs |
  | determinism | two runs → identical CSV |
  | contract | artifact lengths == 100000; CSV schema; validator returns 0 |
- Fixtures use a tiny synthetic candidate set committed under `tests/data/` (≤20 records).

```toml
[tool.pytest.ini_options]
addopts = "-q --strict-markers --cov=src/aptus --cov-report=term-missing --cov-fail-under=85"
testpaths = ["tests"]
```

---

## 9. Pre-commit (local quality gate)

`.pre-commit-config.yaml` runs on every commit so bad code never enters history:
- `ruff check --fix`, `ruff format`
- `mypy src/aptus`
- trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files (block >5 MB)
- `uv lock --check` (lock stays in sync with pyproject)
- (optional) `nbstripout` to keep notebook diffs clean

```bash
uv run pre-commit install        # one-time
uv run pre-commit run --all-files
```

---

## 10. Commits & versioning

- **Conventional Commits** (`feat`, `fix`, `eval`, `docs`, `test`, `chore`, `perf`, `refactor`).
  Full convention + scopes in [06_git_strategy.md](./06_git_strategy.md).
- **SemVer-style tags** per phase milestone (`v0.1`…`v1.0`); submission tags `submission-S1/2/3`.
- Scoring-related commits cite the eval delta (the audit trail).
- `CHANGELOG.md` is generated from conventional-commit history at each tag.

---

## 11. Documentation standards

- Google-style docstrings on every public function/class (enforced by ruff `D`).
- Module headers state purpose + which FR/PRD section they implement.
- Architecture/decisions live in `docs/`; ADR-style notes for any reversible-but-significant choice.
- README always carries the *current* reproduce command and constraint-compliance table.

---

## 12. Definition of "production-ready" for this repo

A module is done only when:
- [ ] fully typed, passes `mypy --strict`
- [ ] ruff clean (lint + format)
- [ ] unit + contract tests, coverage ≥85% for that module
- [ ] structured logging, no `print`, no bare `except`
- [ ] config-driven (no magic numbers; values in `jd_requirements.yaml`)
- [ ] docstrings + module header referencing its requirement
- [ ] runs under `uv run`, reproducible, deterministic where applicable
