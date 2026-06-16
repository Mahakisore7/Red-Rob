# 10 · Containerization & Reproducibility

**Project:** Aptus-R (v5) · **Team:** Code Blooded
The judge runs our pipeline in a sandboxed container with no network. This doc makes that
bullet-proof: a deterministic, offline, resource-bounded image built with `uv`.

---

## 1. Why a container

- The spec sandbox is Linux, offline, CPU-only, ≤16 GB. A pinned image *is* the reproduction
  environment — "works in the container" == "works for the judge".
- `uv` + a committed `uv.lock` make the dependency layer byte-reproducible; the base image pins the
  OS + Python; model files are pinned by SHA-256. Nothing floats.

---

## 2. `Dockerfile` (multi-stage, uv-based)

```dockerfile
# ---- Stage 1: builder — resolve deps into a venv with uv ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Layer-cache deps: copy only lock + pyproject first
COPY pyproject.toml uv.lock ./
# Runtime deps only (no dev, no heavy precompute extra) → lean Phase-B image
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the package itself
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev

# ---- Stage 2: runtime — minimal, offline ----
FROM python:3.11-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app

# non-root user (production hygiene)
RUN useradd --create-home --uid 1000 aptus
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config
COPY rank.py eval.py ./
# artifacts/ and models/ are bind-mounted or COPYed at run time (large, pinned by sha)

USER aptus
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "data/candidates.jsonl.gz", "--out", "submission.csv"]
```

Notes:
- **Multi-stage** keeps the runtime image small (no build tools, no dev deps).
- `UV_PYTHON_DOWNLOADS=never` → the base image's Python is used; no surprise downloads.
- `OMP_NUM_THREADS=1` + single-thread LLM → determinism inside the container.
- Non-root user, no shell tools in the final layer → smaller attack surface, production hygiene.

---

## 3. `.dockerignore`

```
.venv/  __pycache__/  *.pyc
.git/  .github/  notebooks/  sandbox/  tests/
data/*.jsonl.gz          # mounted at run, not baked
models/*.gguf            # mounted at run, pinned by sha
artifacts/*.index        # FAISS rebuilt or mounted
docs/  *.pdf
```

---

## 4. Building & running

```bash
# build the lean offline ranking image
docker build -t aptus-r:1.0 .

# Phase B run — NO network, capped RAM, artifacts+models+data mounted read-only
docker run --rm \
  --network none \
  --memory 16g --cpus 4 \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/out:/app/out" \
  aptus-r:1.0 --candidates data/candidates.jsonl.gz --out out/submission.csv
```

`--network none` is the literal enforcement of NFR-3: if any code tries to reach the network, the
run fails — proving Phase B is offline.

---

## 5. Reproducibility ladder (what's pinned at each layer)

| Layer | Pinned by | Verifies |
|---|---|---|
| OS + system libs | `python:3.11-slim` digest (`@sha256:...`) | identical base |
| Python | base image (`UV_PYTHON_DOWNLOADS=never`) | identical interpreter |
| Dependencies | `uv.lock` (`uv sync --frozen`) | identical wheels |
| Models | `models/MODEL_MANIFEST.json` SHA-256 | identical weights |
| Source | git tag (`v1.0-final`) | identical code |
| Randomness | seeds + 1 thread + temp=0 | identical output |

Pin the base image by digest in the final build:
```dockerfile
FROM python:3.11-slim@sha256:<digest> AS runtime
```

---

## 6. Offline & determinism verification (mirrors CI)

```bash
# 1) determinism inside the container
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts:ro" \
  -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out" \
  aptus-r:1.0 --candidates data/candidates.jsonl.gz --out out/a.csv
# ...repeat to out/b.csv...
diff out/a.csv out/b.csv && echo "deterministic ✓"

# 2) offline already enforced by --network none; a network attempt aborts the run
```

---

## 7. Resource guardrails

- `--memory 16g` matches the hard RAM cap; the process is OOM-killed if it exceeds (we test under it).
- `--cpus 4` simulates a constrained judge box → exercises the adaptive LLM timing gate.
- A `docker stats` capture during a run is saved to `submission_metadata.yaml` (`ram_peak_gb`,
  `wall_clock_sec`) as measured evidence.

---

## 8. Image size budget

| Layer | Approx size |
|---|---|
| python:3.11-slim | ~120 MB |
| runtime deps (faiss, llama-cpp, numpy, pandas, sklearn) | ~700 MB |
| source + config | < 5 MB |
| **image total** | **~0.8 GB** (models + artifacts mounted, not baked) |

Keeping models/artifacts as mounts (not image layers) keeps the image small and lets the judge supply
the dataset path without rebuilding.

---

## 9. Optional: HF Spaces parity (sandbox)

The Streamlit sandbox (Phase 5) uses the same `pyproject` extras (`sandbox`) and the same `src/aptus`
code path. Its Space `Dockerfile`/`requirements` are generated from the same lock so the demo and the
submission can never diverge.
