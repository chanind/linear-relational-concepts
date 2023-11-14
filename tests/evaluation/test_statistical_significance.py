import pytest

from linear_relational_concepts.evaluation.statistical_significance import (
    calculate_statistical_significance,
)


def test_statistical_significance() -> None:
    # test that the results match https://abtestguide.com/calc/

    result = calculate_statistical_significance(
        trials_a=80000,
        successes_a=1600,
        trials_b=78000,
        successes_b=1696,
    )

    assert result.p_value == pytest.approx(0.0154, abs=0.0001)
    assert result.z_score == pytest.approx(2.4233, abs=0.01)
    assert result.is_significant(0.05)
    assert not result.is_significant(0.01)
    assert result.variation_b_wins
    assert not result.variation_a_wins
