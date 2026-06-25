"""``aptus-eval`` — Phase-C evaluation harness (TRD FR-24/25, PHASE_4).

Scores a ranking against the weak-ground-truth gold set: NDCG@10, NDCG@50, MAP,
P@10, the organizer composite, and the honeypot rate. Pure metric math lives in
``aptus.eval_metrics``; this module is the file I/O + reporting glue.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from aptus import eval_metrics as m
from aptus.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def load_gold(path: str | Path) -> dict[str, int]:
    """Load gold_set.csv (columns: candidate_id, relevance) -> {id: relevance}."""
    gold: dict[str, int] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gold[row["candidate_id"].strip()] = int(float(row["relevance"]))
    return gold


def load_ranking(path: str | Path) -> list[str]:
    """Load a submission CSV (or any candidate_id,rank,... CSV) in rank order."""
    rows: list[tuple[int, str]] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((int(row["rank"]), row["candidate_id"].strip()))
    rows.sort(key=lambda r: r[0])
    return [cid for _, cid in rows]


def evaluate_ranking(
    ranked_ids: list[str],
    gold: dict[str, int],
    honeypot_ids: set[str] | None = None,
) -> dict[str, float]:
    """Compute the full metric set for a ranking against the gold relevances."""
    honeypot_ids = honeypot_ids or set()
    relevances = [float(gold.get(cid, 0)) for cid in ranked_ids]
    ideal = [float(v) for v in gold.values()]
    total_relevant = sum(1 for v in gold.values() if v >= m.RELEVANT_THRESHOLD)

    ndcg10 = m.ndcg_at_k(relevances, ideal, 10)
    ndcg50 = m.ndcg_at_k(relevances, ideal, 50)
    p10 = m.precision_at_k(relevances, 10)
    map_ = m.average_precision(relevances, total_relevant)
    honeypot_rate = (
        sum(1 for cid in ranked_ids if cid in honeypot_ids) / len(ranked_ids) if ranked_ids else 0.0
    )
    return {
        "ndcg@10": ndcg10,
        "ndcg@50": ndcg50,
        "map": map_,
        "p@10": p10,
        "composite": m.composite_challenge(ndcg10, ndcg50, map_, p10),
        "honeypot_rate": honeypot_rate,
    }


def _load_honeypots(artifacts_dir: str | Path) -> set[str]:
    path = Path(artifacts_dir) / "honeypot_ids.json"
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set()


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aptus-eval`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aptus-eval",
        description="Evaluate a submission against the weak ground-truth gold set.",
    )
    parser.add_argument("--submission", required=True, help="Path to submission.csv.")
    parser.add_argument("--gold", default="eval/gold_set.csv", help="Path to gold_set.csv.")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="For honeypot_ids.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``aptus-eval`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    gold = load_gold(args.gold)
    ranked = load_ranking(args.submission)
    metrics = evaluate_ranking(ranked, gold, _load_honeypots(args.artifacts_dir))
    print(f"Evaluated {len(ranked)} ranked vs {len(gold)} gold labels:")
    for name, value in metrics.items():
        print(f"  {name:14s}: {value:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
