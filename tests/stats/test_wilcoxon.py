import pandas as pd
import pytest

from labicompare.core.data import EvaluationData
from labicompare.stats.wilcoxon import wilcoxon_test


def test_wilcoxon_auto_select_models() -> None:
    """Test if function select the two unique models automatically."""
    df = pd.DataFrame(
        {
            "Baseline": [0.80, 0.82, 0.85],
            "New_Model": [0.85, 0.88, 0.90],
        }
    )

    eval_data = EvaluationData(df)
    w_stat, p_value = wilcoxon_test(eval_data)

    assert w_stat >= 0.0
    assert 0.0 <= p_value <= 1.0


def test_wilcoxon_specify_models() -> None:
    """Test the comparison of two specified models."""
    df = pd.DataFrame(
        {
            "Model_A": [0.9, 0.8, 0.85],
            "Model_B": [0.7, 0.6, 0.75],
            "Model_C": [0.5, 0.4, 0.55],
        }
    )
    eval_data = EvaluationData(df)
    
    w_stat, p_value = wilcoxon_test(eval_data, model_1="Model_A", model_2="Model_C")
    assert w_stat >= 0.0
    assert p_value < 0.5


def test_wilcoxon_errors() -> None:
    """Test if validation errors are correctly."""
    df = pd.DataFrame(
        {"Model_A": [0.9, 0.8], "Model_B": [0.7, 0.6], "Model_C": [0.5, 0.4]}
    )
    eval_data = EvaluationData(df)
    
    with pytest.raises(ValueError, match="You must specify"):
        wilcoxon_test(eval_data)

    with pytest.raises(ValueError, match="Models not found"):
        wilcoxon_test(eval_data, model_1="Model_A", model_2="Model_X")

    with pytest.raises(ValueError, match="must be different"):
        wilcoxon_test(eval_data, model_1="Model_A", model_2="Model_A")

