# Phase 0 · Foundations

**Owner:** Mahakisore · **Duration:** ~Day 1 (first half) · **Git tag at DoD:** `v0.1-foundations`
**Goal:** A runnable repo, a clean environment, a real understanding of the data, and the v2-Max
plumbing (schema + gate) reused so we don't re-derive solved problems.

---

## 0. Why this phase exists
Two of v4's flaws are closed before any scoring code is written:
- **F9** (starts from zero code) → reuse v2-Max's tested `schema.py` / `honeypot.py` / `config.py`.
- **F5** (assessment scores under-used) → adopt `skill_index()` from day one.
EDA also de-risks every later weight choice by grounding it in real distributions.

---

## 1. Tasks

### T0.1 — Repo & environment (uv, src-layout)
```bash
mkdir aptus-r && cd aptus-r && git init && git branch -M main
echo "3.11" > .python-version
uv python install 3.11
# author pyproject.toml (see 08_dev_setup.md §3) with src-layout + [project.scripts], then:
uv sync --all-extras --dev          # creates .venv, writes uv.lock
uv run pre-commit install
uv run python -c "import faiss, rank_bm25, llama_cpp, structlog; print('stack ok')"
```
Create the repo skeleton from [07_engineering_standards.md §2](../docs/07_engineering_standards.md)
(src-layout). Add `.gitignore` (`.venv/`, `__pycache__/`, `data/*.jsonl.gz`, `models/*.gguf`,
`artifacts/*.index`), `.dockerignore`, `.editorconfig`. Commit `pyproject.toml` + `uv.lock` from the
first real commit (NFR-7). Configure ruff/mypy/pytest in `pyproject.toml` per the standards doc.

### T0.2 — Copy & adapt v2-Max code (G3)
Bring over and lightly adapt:
- `src/aptus/config.py` — `REFERENCE_DATE=2026-06-01`, paths, `WEIGHTS=load(jd_requirements.yaml)`.
- `src/aptus/schema.py` — dataclasses + `as_of_months()` + `skill_index()` + `full_text()`.
- `src/aptus/honeypot.py` — extend the 5 v2-Max checks to the **6-rule v5 set** (add FR-7c
  too-many-experts, FR-7d perfect-no-verify, FR-7e keyword-stuffer, FR-7f assessment-contradiction;
  keep FR-7a/b). All thresholds in `jd_requirements.yaml`.
- `config/jd_requirements.yaml`, `concept_thesaurus.yaml`, `title_taxonomy.yaml` — copy and confirm
  every weight matches [04_data_model.md](../docs/04_data_model.md).

### T0.3 — Exploratory data analysis (`notebooks/eda.ipynb`)
Confirm/refresh the pool facts that drive every weight. Produce:
- Title distribution → tier pyramid (non-tech vs general SWE vs data/analytics vs ML/AI vs rare senior).
- Skill frequency: identify Tier-B buzzwords (~4–5k each), Tier-C deep skills (~1–1.4k), Tier-D
  plain-language (1–7 occurrences).
- Company landscape: confirm TCS/Infosys/Wipro ubiquity vs rare product firms (Zoho/Swiggy/CRED…).
- `redrob_signals` distributions: median `last_active` (~122d), `recruiter_response_rate` (~0.44),
  `-1` sentinel rates for github/offer.
- Honeypot scan: count candidates each of the 6 rules flags; eyeball 5 examples per rule.

### T0.4 — Honeypot gate validation
Run the gate on the full 100K. Assert it flags the known cases:
- `CAND_0007353` (FR-7a duration-vs-dates + FR-7b career-math),
- `CAND_0004989` (salary sanity — keep as FR-7 optional),
- the 21 expert-zero-duration candidates.
Write `tests/test_honeypot.py` covering one candidate per rule.

### T0.5 — JD as structured input
Confirm `jd_requirements.yaml` captures: component weights, 6 skill concepts + weights, all
modifiers, all penalties, integrity thresholds, tiebreak. Each line annotated with its JD sentence.

---

## 2. Deliverables
- [ ] Repo skeleton (src-layout) + `pyproject.toml` + committed `uv.lock` + `.gitignore`/`.dockerignore`
- [ ] Toolchain wired: ruff + mypy + pytest configs in `pyproject.toml`; `.pre-commit-config.yaml` installed
- [ ] `.github/workflows/ci.yml` green on first push (lint/type/test gate)
- [ ] `config.py`, `schema.py` (with `skill_index`), `honeypot.py` (6 rules) — typed, imported & tested
- [ ] `config/*.yaml` complete and annotated
- [ ] `notebooks/eda.ipynb` with the pool facts above (nbstripout clean)
- [ ] `tests/test_honeypot.py` green; coverage gate configured (≥85%)
- [ ] `honeypot_ids.json` count is sane (~70–90, not thousands)

---

## 3. Definition of Done
- `make check` (ruff + mypy + pytest) passes locally and in CI.
- `uv run pytest tests/test_honeypot.py` passes; coverage gate active.
- Gate flags all known honeypots; flag count in the expected ~80 range.
- EDA notebook committed; every weight in `jd_requirements.yaml` has a data/JD justification.
- `uv run python -c "from aptus.schema import skill_index"` works.
- pre-commit installed; CI green on `main`.

**Git:** merge `phase-0` → `main`, tag `v0.1-foundations`.

---

## 4. Risks
| Risk | Mitigation |
|---|---|
| v2-Max code assumes slightly different schema keys | Adapt in `from_dict`; cover with a parse smoke test |
| Honeypot rules too aggressive (flag real candidates) | Validate against EDA; tune thresholds in YAML, not code |
| EDA reveals new trap pattern | Add a rule now (cheap) rather than discovering it at submission |
