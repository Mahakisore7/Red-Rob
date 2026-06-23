"""5-signal composite scoring tests (S1, composite, modifiers, penalties)."""

from __future__ import annotations

from aptus import signals


def _feat(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "s2_career_arc": 1.0,
        "s3_behavioral": 1.0,
        "s4_recency": 1.0,
        "s5_intent": 1.0,
        "notice_mod": 1.0,
        "location_mod": 1.0,
        "salary_mod": 1.0,
        "work_mod": 1.0,
        "is_consulting_only": False,
        "is_title_chaser": False,
        "has_product_exp": True,
        "is_honeypot": False,
    }
    base.update(overrides)
    return base


def test_s1_semantic_stretch() -> None:
    assert signals.s1_semantic(0.30) == 0.0  # at floor
    assert signals.s1_semantic(0.20) == 0.0  # below floor -> clipped
    assert abs(signals.s1_semantic(0.95) - 1.0) < 1e-9  # floor + range (float rounding)
    assert signals.s1_semantic(2.0) == 1.0  # above -> clipped
    assert abs(signals.s1_semantic(0.625) - 0.5) < 1e-9  # midpoint


def test_composite_weights_sum_to_one() -> None:
    # all signals = 1.0 -> composite = sum of weights = 1.0
    assert abs(signals.composite(1.0, _feat()) - 1.0) < 1e-9


def test_signals_in_unit_range_for_composite() -> None:
    val = signals.composite(0.5, _feat(s2_career_arc=0.4, s3_behavioral=0.6))
    assert 0.0 <= val <= 1.0


def test_modifier_product() -> None:
    assert signals.modifier_product(_feat()) == 1.0
    assert abs(signals.modifier_product(_feat(notice_mod=0.85, work_mod=0.9)) - 0.765) < 1e-9


def test_penalty_product_stacks() -> None:
    assert signals.penalty_product(_feat()) == 1.0
    assert signals.penalty_product(_feat(is_consulting_only=True)) == 0.60
    assert signals.penalty_product(_feat(has_product_exp=False)) == 0.70
    # stacking consulting + title_chaser + no product = 0.60 * 0.75 * 0.70
    stacked = signals.penalty_product(
        _feat(is_consulting_only=True, is_title_chaser=True, has_product_exp=False)
    )
    assert abs(stacked - 0.60 * 0.75 * 0.70) < 1e-9


def test_honeypot_mult() -> None:
    assert signals.honeypot_mult(False) == 1.0
    assert signals.honeypot_mult(True) == 0.05


def test_final_score_full_formula() -> None:
    feat = _feat(notice_mod=0.85, is_consulting_only=True, is_honeypot=True)
    expected = 1.0 * 0.85 * 0.60 * 0.05  # composite(1) * mods * penalty * honeypot
    assert abs(signals.final_score(1.0, feat) - expected) < 1e-9


def test_honeypot_crushes_score() -> None:
    clean = signals.final_score(1.0, _feat())
    trapped = signals.final_score(1.0, _feat(is_honeypot=True))
    assert trapped < clean
    assert abs(trapped - clean * 0.05) < 1e-9
