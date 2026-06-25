"""T4 — ablation + Decision Rule 1 over the gold set (PHASE_4 / docs/05 §3-§4).

Ranks the gold candidates by four methods (naive, title-only, composite,
composite+LLM) and scores each with the eval metrics, then evaluates the LLM
blend weight w in {1.0, 0.70, 0.40}. Writes eval/eval_report.md.

The LLM is run only over the top-30 of the composite ranking (mirrors the live
pipeline) so this stays ~30 CPU calls. Run:
    python scripts/run_ablation.py --artifacts-dir artifacts --gold eval/gold_set.csv
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import sys
from pathlib import Path

import orjson

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aptus import signals  # noqa: E402
from aptus.cli.eval import evaluate_ranking, load_gold  # noqa: E402
from aptus.config import LLM_CFG  # noqa: E402
from aptus.honeypot import _AIML_SKILL_FRAGMENTS  # noqa: E402
from aptus.llm_reranker import build_prompt, parse_response  # noqa: E402
from aptus.retriever import Artifacts, load_artifacts  # noqa: E402
from scripts.build_gold_set import compute_s1_all, title_tier  # noqa: E402

_TIER_RANK = {"tier_1": 5, "tier_2": 4, "tier_3": 3, "tier_4": 2, "tier_5": 1, "unmapped": 0}


def _raw_records(path: Path, wanted: set[str]) -> dict[str, dict]:  # type: ignore[type-arg]
    """Stream the raw data and collect records for the wanted candidate ids."""
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, dict] = {}  # type: ignore[type-arg]
    with opener(path, "rb") as fh:  # type: ignore[operator]
        for line in fh:
            if not line.strip():
                continue
            rec = orjson.loads(line)
            if rec["candidate_id"] in wanted:
                out[rec["candidate_id"]] = rec
                if len(out) == len(wanted):
                    break
    return out


def rank_naive(gold_cids: list[str], raw: dict[str, dict]) -> list[str]:  # type: ignore[type-arg]
    """Naive baseline: AI-skill-count * recruiter_response_rate (the sample trap)."""

    def score(cid: str) -> float:
        rec = raw.get(cid)
        if not rec:
            return 0.0
        skills = rec.get("skills", [])
        ai = sum(
            1 for s in skills if any(f in s.get("name", "").lower() for f in _AIML_SKILL_FRAGMENTS)
        )
        rr = float(rec.get("redrob_signals", {}).get("recruiter_response_rate", 0.0))
        return ai * rr

    return sorted(gold_cids, key=lambda c: (-score(c), c))


def rank_title(gold_cids: list[str], art: Artifacts) -> list[str]:
    """Title-only baseline: rank by taxonomy tier."""
    return sorted(
        gold_cids,
        key=lambda c: (-_TIER_RANK[title_tier(str(art.facts.loc[c, "current_title"]))], c),
    )


def composite_scores(
    gold_cids: list[str], art: Artifacts, s1: dict[str, float]
) -> dict[str, float]:
    """Final composite score per gold candidate."""
    return {c: signals.final_score(s1[c], art.features.loc[c].to_dict()) for c in gold_cids}


def llm_fit_topk(gold_cids_by_comp: list[str], art: Artifacts, k: int) -> dict[str, float]:
    """Run the real LLM over the top-k composite candidates; return fit (0-100)."""
    from aptus.llm_reranker import PhiReranker

    gen = PhiReranker().generate
    fits: dict[str, float] = {}
    for cid in gold_cids_by_comp[:k]:
        fact = art.facts.loc[cid].to_dict()
        fact["candidate_id"] = cid
        parsed = parse_response(gen(build_prompt(fact)))
        if parsed is not None:
            fits[cid] = float(parsed["fit_score"])
    return fits


def _row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {metrics['ndcg@10']:.3f} | {metrics['ndcg@50']:.3f} | "
        f"{metrics['map']:.3f} | {metrics['p@10']:.3f} | {metrics['composite']:.3f} | "
        f"{metrics['honeypot_rate']:.2f} |"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the ablation + Decision Rule 1 and write eval/eval_report.md."""
    p = argparse.ArgumentParser(prog="run_ablation", description=__doc__)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--gold", default="eval/gold_set.csv")
    p.add_argument("--candidates", default="data/candidates.jsonl")
    p.add_argument("--out", default="eval/eval_report.md")
    args = p.parse_args(argv)

    art = load_artifacts(args.artifacts_dir)
    gold = load_gold(args.gold)
    gold_cids = list(gold)
    hp = art.honeypot_ids

    s1_arr = compute_s1_all(art)
    s1 = {art.id_map[i]: float(s1_arr[i]) for i in range(art.index.ntotal)}

    print("ranking naive / title / composite ...")
    raw = _raw_records(Path(args.candidates), set(gold_cids))
    naive = rank_naive(gold_cids, raw)
    title = rank_title(gold_cids, art)
    comp = composite_scores(gold_cids, art, s1)
    comp_rank = sorted(gold_cids, key=lambda c: (-comp[c], c))

    print("running LLM over top-30 of composite (CPU) ...")
    k = int(LLM_CFG["default_top_k"])
    fits = llm_fit_topk(comp_rank, art, k)

    def blended_rank(w: float) -> list[str]:
        def sc(c: str) -> float:
            return signals.blend(comp[c], fits[c], w) if c in fits else comp[c]

        return sorted(gold_cids, key=lambda c: (-sc(c), c))

    results = {
        "naive (AI-count x response)": evaluate_ranking(naive, gold, hp),
        "title-only (tier)": evaluate_ranking(title, gold, hp),
        "composite (S1-S5, w=1.0)": evaluate_ranking(comp_rank, gold, hp),
        "composite+LLM (w=0.70)": evaluate_ranking(blended_rank(0.70), gold, hp),
        "composite+LLM (w=0.40)": evaluate_ranking(blended_rank(0.40), gold, hp),
    }

    today = datetime.date.today().isoformat()
    lines = [
        "# Eval Report (PHASE_4)",
        "",
        f"_Generated {today}. Gold set: {len(gold)} **weak/heuristic** labels "
        "(S1-correlated; hand-correct before trusting). LLM run over top-30 of composite._",
        "",
        "| ranker | NDCG@10 | NDCG@50 | MAP | P@10 | composite | honeypot |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_row(name, mtr) for name, mtr in results.items()]
    dr1 = max(
        [
            ("1.00", results["composite (S1-S5, w=1.0)"]["ndcg@10"]),
            ("0.70", results["composite+LLM (w=0.70)"]["ndcg@10"]),
            ("0.40", results["composite+LLM (w=0.40)"]["ndcg@10"]),
        ],
        key=lambda x: x[1],
    )
    lines += [
        "",
        f"## Decision Rule 1\nBest NDCG@10 at **w={dr1[0]}** ({dr1[1]:.3f}). "
        "Ship the smallest (1-w) that maximises holdout NDCG@10; tie -> prefer composite (w=1.0).",
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for name, mtr in results.items():
        print(f"  {name:32s} NDCG@10={mtr['ndcg@10']:.3f} composite={mtr['composite']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
