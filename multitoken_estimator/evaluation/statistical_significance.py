from dataclasses import dataclass

import numpy as np
import scipy.stats as stats


@dataclass
class StatisticalSignificanceResult:
    p_value: float
    z_score: float

    def is_significant(self, significance_threshold: float = 0.05) -> bool:
        return self.p_value < significance_threshold

    @property
    def variation_a_wins(self) -> bool:
        return self.z_score < 0

    @property
    def variation_b_wins(self) -> bool:
        return self.z_score > 0


def calculate_statistical_significance(
    trials_a: int,
    successes_a: int,
    trials_b: int,
    successes_b: int,
) -> StatisticalSignificanceResult:
    """
    Calculate statistical significance between two trials, using 2-sided z-test
    """
    p1 = successes_a / trials_a
    p2 = successes_b / trials_b

    # Pooled proportion
    p_pool = (successes_a + successes_b) / (trials_a + trials_b)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / trials_a + 1 / trials_b))

    # Z-score
    z_score = (p2 - p1) / se

    # P-value
    p_value = 2 * stats.norm.cdf(-np.abs(z_score))

    return StatisticalSignificanceResult(p_value=p_value, z_score=z_score)
