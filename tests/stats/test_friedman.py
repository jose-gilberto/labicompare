import pandas as pd
import pytest

from labicompare.core.data import EvaluationData
from labicompare.stats.friedman import friedman_test


def friedman_test_computation() -> None:
    """Test if friedman stats and p-value are calculated correctly."""
    df = pd.DataFrame(
        {
            "Model_A": [0.9, 0.8, 0.85, 0.92],
            "Model_B": [0.7, 0.6, 0.75, 0.72],
            "Model_C": [0.5, 0.4, 0.55, 0.60],
        }
    )

    eval_data = EvaluationData(df)
    f_stat, p_value = friedman_test(eval_data)

    assert f_stat > 0.0
    assert p_value > 0.05


def test_friedman_insufficient_models() -> None:
    """Friedman test must fail if we dont have at least 3 models."""
    df = pd.DataFrame(
        {
            "Model_A": [0.9, 0.8, 0.85, 0.92],
            "Model_B": [0.7, 0.6, 0.75, 0.72],
        }
    )

    eval_data = EvaluationData(df)

    with pytest.raises(ValueError, match="requires at least 3 models"):
        friedman_test(eval_data)
