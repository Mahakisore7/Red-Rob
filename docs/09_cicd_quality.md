# 09 · CI/CD & Quality Gates

**Project:** Aptus-R (v5) · **Team:** Code Blooded
Automated gates so `main` is always green, always reproducible, and the git history is trustworthy
evidence for Stage 3.

---

## 1. The quality gate (what must pass before merge)

| Gate | Tool | Blocks merge? |
|---|---|---|
| Format | `ruff format --check` | yes |
| Lint | `ruff check` | yes |
| Types | `mypy --strict src/aptus` | yes |
| Tests + coverage ≥85% | `pytest --cov-fail-under=85` | yes |
| Lock in sync | `uv lock --check` | yes |
| Determinism | `test_determinism.py` | yes (pre-tag) |
| No large files | pre-commit `check-added-large-files` | yes |
| Offline ranking | offline smoke test | yes (pre-tag) |

Same gate runs in three places: **pre-commit** (fast subset, local), **CI** (full, on every push/PR),
and **pre-tag** (adds determinism + offline + timing). Defense in depth.

---

## 2. `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.5
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.1
    hooks:
      - id: mypy
        additional_dependencies: [numpy, pandas-stubs, types-PyYAML]
        files: ^src/aptus/
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=5120]        # block >5 MB (no artifacts/models in git)
      - id: check-merge-conflict
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout            # clean notebook diffs
  - repo: local
    hooks:
      - id: uv-lock-check
        name: uv lock in sync
        entry: uv lock --check
        language: system
        pass_filenames: false
```

---

## 3. GitHub Actions — `.github/workflows/ci.yml`

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request: {}

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with: { enable-cache: true }
      - run: uv python install 3.11
      - run: uv sync --all-extras --dev --frozen
      - name: Lock in sync
        run: uv lock --check
      - name: Format
        run: uv run ruff format --check .
      - name: Lint
        run: uv run ruff check .
      - name: Types
        run: uv run mypy src/aptus
      - name: Tests + coverage
        run: uv run pytest
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with: { name: coverage, path: htmlcov/ }
```

CI runs on Ubuntu (judge-like Linux). `setup-uv` caches the resolver → CI finishes in ~1–2 min.

---

## 4. Determinism & offline job — `.github/workflows/repro.yml`

Runs on demand and before every submission tag. Uses a tiny committed fixture dataset so it needs no
large artifacts.

```yaml
name: repro
on:
  workflow_dispatch: {}
  push: { tags: ['v*', 'submission-*'] }

jobs:
  determinism:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - name: Build fixture artifacts
        run: uv run aptus-precompute --candidates tests/data/mini.jsonl.gz --out-dir /tmp/art
      - name: Rank twice
        run: |
          uv run aptus-rank --candidates tests/data/mini.jsonl.gz --out /tmp/a.csv
          uv run aptus-rank --candidates tests/data/mini.jsonl.gz --out /tmp/b.csv
      - name: Assert byte-identical
        run: diff /tmp/a.csv /tmp/b.csv && echo "deterministic ✓"
      - name: Offline check (no network during rank)
        run: |
          # unshare network namespace to prove Phase B needs no network
          sudo unshare -n uv run aptus-rank --candidates tests/data/mini.jsonl.gz --out /tmp/c.csv
          echo "offline ✓"
```

The `unshare -n` step is the strongest possible proof of NFR-3: ranking succeeds with the network
namespace removed.

---

## 5. Branch protection (GitHub settings)

- `main` requires: PR + passing `ci` workflow + ≥1 review (teammate).
- No direct pushes to `main` (except the initial `init` commit).
- Linear history (squash or rebase merges) to keep the story readable for Stage 3.

---

## 6. Release / tag automation

On a `v*` tag:
1. `repro.yml` runs (determinism + offline).
2. `CHANGELOG.md` is regenerated from conventional commits since the last tag.
3. `uv export` regenerates `requirements.txt`; the produced `submission.csv` SHA-256 is recorded in
   the tag annotation (per [06_git_strategy.md](./06_git_strategy.md)).

```bash
git tag -a v0.3-safety -m "safety submission: valid non-LLM CSV
csv_sha256: $(sha256sum submission.csv | cut -d' ' -f1)"
git push origin v0.3-safety
```

---

## 7. What CI proves to judges (Stage 3)

- **Reproducible:** `uv sync --frozen` + lockfile → identical deps everywhere.
- **Deterministic:** the `diff` job proves same-input→same-output.
- **Offline-capable:** the `unshare -n` job proves Phase B needs no network.
- **Quality:** typed, linted, ≥85% covered — not a weekend script.
- **Authentic iteration:** green history, conventional commits, phase tags.
