"""T4 — bootstrap a Weak Ground Truth gold set (PHASE_4 / docs/05 §1).

Produces ``eval/gold_set.csv`` (candidate_id, relevance 0-3) by stratified sampling
+ a transparent rule-based auto-labeler over the Phase-A artifacts. These are
**weak/heuristic** labels meant as a starting point — the team should spot-check
and correct them (esp. the 2-vs-3 boundary) before trusting the eval numbers.

Run:
    python scripts/build_gold_set.py --artifacts-dir artifacts --n 180 --out eval/gold_set.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402

from aptus import signals  # noqa: E402
from aptus.retriever import Artifacts, load_artifacts  # noqa: E402

SEED = 42
_TAXONOMY = yaml.safe_load((_ROOT / "config" / "title_taxonomy.yaml").read_text(encoding="utf-8"))


def title_tier(title: str) -> str:
    """Map a title to a taxonomy tier (or 'unmapped')."""
    t = (title or "").lower()
    for tier_name, block in _TAXONOMY["tiers"].items():
        if any(ex in t for ex in block["examples"]):
            return tier_name
    return "unmapped"


def compute_s1_all(art: Artifacts) -> np.ndarray:  # type: ignore[type-arg]
    """S1 semantic value for every candidate (cosine of its vector vs the JD)."""
    vecs = art.index.reconstruct_n(0, art.index.ntotal)
    cos = vecs @ art.jd_vector.astype(np.float32)
    return np.array([signals.s1_semantic(float(c)) for c in cos], dtype=np.float32)


def heuristic_label(feat: dict, s1: float, tier: str) -> int:  # type: ignore[type-arg]
    """Transparent 0-3 weak label (docs/05 §1.2 rubric).

    Semantic-led (titles are too sparsely mapped to gate on). NOTE: this reuses S1,
    which the composite also uses, so the labels are *not independent* of the ranker
    -- treat ablation numbers as indicative, and hand-correct before trusting them.
    """
    if bool(feat["is_honeypot"]) or tier == "tier_5":
        return 0
    product = bool(feat["has_product_exp"])
    consulting = bool(feat["is_consulting_only"])
    active = float(feat["s4_recency"]) > 0.5
    s2 = float(feat["s2_career_arc"])

    if s1 >= 0.60 and s2 >= 0.55 and product and active and not consulting:
        return 3  # right trajectory + product + available
    if (s1 >= 0.55 and s2 >= 0.50) or tier in ("tier_1", "tier_2"):
        return 2  # strong technically (maybe with a real gap)
    if s1 >= 0.45 or tier in ("tier_3", "tier_4"):
        return 1  # transferable / adjacent
    return 0


def stratified_sample(cands: list[str], s1: dict[str, float], hp: set[str], n: int) -> list[str]:
    """Stratify by honeypot + S1-rank bands so all relevance levels are represented."""
    rng = np.random.default_rng(SEED)
    honeypots = [c for c in cands if c in hp]
    clean = sorted((c for c in cands if c not in hp), key=lambda c: s1[c], reverse=True)
    total = len(clean)

    # rank-based bands over the clean pool: genuine top -> strong -> adjacent -> general
    strata = [clean[0:500], clean[500:3000], clean[3000:15000], clean[15000:total]]
    quotas = [40, 40, 35, 35]
    quota_hp = min(len(honeypots), 30)

    picked: list[str] = (
        list(rng.choice(honeypots, size=quota_hp, replace=False)) if honeypots else []
    )
    for band, q in zip(strata, quotas, strict=False):
        take = min(q, len(band))
        if take:
            picked.extend(rng.choice(band, size=take, replace=False))
    return sorted(set(picked))


def build(artifacts_dir: str, n: int) -> list[tuple[str, int]]:
    """Sample + label; return [(candidate_id, relevance)]."""
    art = load_artifacts(artifacts_dir)
    cands = [art.id_map[i] for i in range(art.index.ntotal)]
    s1_arr = compute_s1_all(art)
    s1 = {art.id_map[i]: float(s1_arr[i]) for i in range(len(cands))}

    sample = stratified_sample(cands, s1, art.honeypot_ids, n)
    rows: list[tuple[str, int]] = []
    for cid in sample:
        feat = art.features.loc[cid].to_dict()
        tier = title_tier(str(art.facts.loc[cid, "current_title"]))
        rows.append((cid, heuristic_label(feat, s1[cid], tier)))
    return rows


def main(argv: list[str] | None = None) -> int:
    """Build the gold set and write the CSV."""
    p = argparse.ArgumentParser(prog="build_gold_set", description=__doc__)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--n", type=int, default=180)
    p.add_argument("--out", default="eval/gold_set.csv")
    args = p.parse_args(argv)

    rows = build(args.artifacts_dir, args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("candidate_id,relevance\n")
        for cid, rel in rows:
            fh.write(f"{cid},{rel}\n")

    dist = {r: sum(1 for _, x in rows if x == r) for r in (0, 1, 2, 3)}
    print(f"wrote {out} with {len(rows)} labels; distribution {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
