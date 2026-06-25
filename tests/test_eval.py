"""Eval metric tests against hand-computed fixtures (docs/05 §2)."""

from __future__ import annotations

import math

from aptus import eval_metrics as m


def test_dcg_hand_computed() -> None:
    # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5
    expected = 3 / 1 + 2 / math.log2(3) + 1 / 2
    assert abs(m.dcg([3, 2, 1], 3) - expected) < 1e-9


def test_ndcg_perfect_and_reversed() -> None:
    assert abs(m.ndcg_at_k([3, 2, 1], [3, 2, 1], 3) - 1.0) < 1e-9
    ranked = m.dcg([1, 2, 3], 3)
    ideal = m.dcg([3, 2, 1], 3)
    assert abs(m.ndcg_at_k([1, 2, 3], [3, 2, 1], 3) - ranked / ideal) < 1e-9


def test_ndcg_zero_ideal() -> None:
    assert m.ndcg_at_k([0, 0], [0, 0], 2) == 0.0


def test_precision_at_k() -> None:
    assert m.precision_at_k([3, 2, 1, 0], 4) == 0.5  # two of four are >=2
    assert m.precision_at_k([3, 3, 3], 3) == 1.0
    assert m.precision_at_k([1, 1], 2) == 0.0


def test_average_precision_hand_computed() -> None:
    # hits at positions 1 and 3: (1/1 + 2/3) / 2
    expected = (1.0 + 2.0 / 3.0) / 2
    assert abs(m.average_precision([2, 0, 2], total_relevant=2) - expected) < 1e-9


def test_average_precision_no_hits() -> None:
    assert m.average_precision([0, 1, 1], total_relevant=2) == 0.0


def test_composite_challenge() -> None:
    val = m.composite_challenge(0.8, 0.7, 0.6, 0.5)
    assert abs(val - (0.5 * 0.8 + 0.3 * 0.7 + 0.15 * 0.6 + 0.05 * 0.5)) < 1e-9


def test_evaluate_ranking_perfect() -> None:
    from aptus.cli.eval import evaluate_ranking

    gold = {"A": 3, "B": 2, "C": 1}
    res = evaluate_ranking(["A", "B", "C"], gold, honeypot_ids={"C"})
    assert abs(res["ndcg@10"] - 1.0) < 1e-9
    assert abs(res["map"] - 1.0) < 1e-9
    assert abs(res["p@10"] - 0.2) < 1e-9  # 2 relevant / 10
    assert abs(res["honeypot_rate"] - 1 / 3) < 1e-9


def test_load_gold_and_ranking(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from aptus.cli.eval import load_gold, load_ranking

    gold_csv = tmp_path / "gold.csv"
    gold_csv.write_text(
        "candidate_id,relevance\nCAND_0000001,3\nCAND_0000002,0\n", encoding="utf-8"
    )
    sub_csv = tmp_path / "sub.csv"
    sub_csv.write_text(
        "candidate_id,rank,score,reasoning\nCAND_0000002,2,0.5,b\nCAND_0000001,1,0.9,a\n",
        encoding="utf-8",
    )
    assert load_gold(gold_csv) == {"CAND_0000001": 3, "CAND_0000002": 0}
    assert load_ranking(sub_csv) == ["CAND_0000001", "CAND_0000002"]  # sorted by rank
