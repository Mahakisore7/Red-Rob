"""Runtime configuration loader for Aptus-R v5.

Reads ``config/jd_requirements.yaml`` once at import time and exposes typed
constants. All magic numbers live in the YAML (FR-12 / G2 graft).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR: Path = REPO_ROOT / "config"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"

_JD_YAML: Path = CONFIG_DIR / "jd_requirements.yaml"
_THESAURUS_YAML: Path = CONFIG_DIR / "concept_thesaurus.yaml"
_TAXONOMY_YAML: Path = CONFIG_DIR / "title_taxonomy.yaml"

# ---------------------------------------------------------------------------
# Load raw YAML once
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict from {path}, got {type(data)}")
    return data


CFG: dict[str, Any] = _load_yaml(_JD_YAML)

# ---------------------------------------------------------------------------
# Typed constants derived from CFG
# ---------------------------------------------------------------------------

#: Fixed reference date for all recency calculations (docs/04_data_model.md).
REFERENCE_DATE: datetime.date = datetime.date.fromisoformat(CFG["reference_date"])

#: Global RNG seed — keeps FAISS tie-breaks and LLM deterministic (NFR-5).
SEED: int = int(CFG["seed"])

#: Signal weights dict, e.g. {"s1_semantic": 0.30, ...}
SIGNAL_WEIGHTS: dict[str, float] = CFG["signal_weights"]

#: Honeypot gate thresholds
HONEYPOT_RULES: dict[str, Any] = CFG["honeypot_rules"]

#: Honeypot score multiplier (0.05) and max allowed in top-100
HONEYPOT_CFG: dict[str, Any] = CFG["honeypot"]

#: JD modifiers block
MODIFIERS: dict[str, Any] = CFG["modifiers"]

#: Disqualifier penalties block
PENALTIES: dict[str, Any] = CFG["penalties"]

#: LLM rerank block
LLM_CFG: dict[str, Any] = CFG["llm"]

#: Retrieval block (FAISS/BM25/RRF pool sizes)
RETRIEVAL_CFG: dict[str, Any] = CFG["retrieval"]

#: Text builder caps
TEXT_CFG: dict[str, Any] = CFG["text_builder"]
