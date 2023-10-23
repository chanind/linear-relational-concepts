from multitoken_estimator.training.ObjectMappingSampleReweighter import (
    _calc_reweighting,
)


def test_calc_reweighting_does_nothing_if_classes_are_balanced() -> None:
    assert _calc_reweighting(0.5, 2) == 1.0
    assert _calc_reweighting(0.25, 4) == 1.0
    assert _calc_reweighting(0.2, 5) == 1.0
    assert _calc_reweighting(0.1, 10) == 1.0


def test_calc_reweighting_increases_weight_for_uncommon_classes() -> None:
    assert _calc_reweighting(0.25, 2) == 2.0
    assert _calc_reweighting(0.1, 5) == 2.0


def test_calc_reweighting_decreases_weight_for_common_classes() -> None:
    assert _calc_reweighting(0.8, 2) == 0.625
