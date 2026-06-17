# Task shortcuts wrapping uv. See docs/08_dev_setup.md §8.
# (On Windows, run these via Git Bash / WSL, or call the uv commands directly.)
.PHONY: setup fmt lint type test cov precompute rank eval check repro

CANDIDATES ?= Dataset/India_runs_data_and_ai_challenge/candidates.jsonl

setup:       ; uv sync --all-extras --dev && uv run pre-commit install
fmt:         ; uv run ruff format .
lint:        ; uv run ruff check --fix .
type:        ; uv run mypy src/aptus
test:        ; uv run pytest
cov:         ; uv run pytest --cov-report=html
check:       lint type test          ## the gate CI also runs
precompute:  ; uv run aptus-precompute --candidates $(CANDIDATES)
rank:        ; uv run aptus-rank --candidates $(CANDIDATES) --out submission.csv
eval:        ; uv run aptus-eval --submission submission.csv --gold eval/gold_set.csv
repro:       precompute rank eval     ## full pipeline end-to-end
