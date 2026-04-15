import pandas as pd

from labicompare.core.data import EvaluationData
from labicompare.stats.posthoc import adjust_holm, posthoc_wilcoxon


def test_adjust_holm_computation() -> None:
    """Test if Bonferroni-Holm correction is calculated correctly."""
    p_values = [0.01, 0.04, 0.03]
    
    # Ascending: 0.01 (multiplies by 3) = 0.03
    # 0.03 (times 2) = 0.06
    # 0.04 (times 1) = 0.04 -> as has to be not-descending, increases to 0.06
    expected = [0.03, 0.06, 0.06]
    
    adj = adjust_holm(p_values)
    
    # Use round to avoid floating point problems with Python
    adj_rounded = [round(p, 4) for p in adj]
    assert adj_rounded == expected


def test_posthoc_wilcoxon_matrix_shape_and_symmetry() -> None:
    """Test if generated matrix has to correct size and its symmetric."""
    df = pd.DataFrame(
        {
            "Model_A": [0.9, 0.8, 0.85, 0.95],
            "Model_B": [0.7, 0.6, 0.75, 0.70],
            "Model_C": [0.5, 0.4, 0.55, 0.50],
        }
    )
    data = EvaluationData(df)
    result = posthoc_wilcoxon(data)

    assert result.shape == (3, 3)
    assert result.loc["Model_A", "Model_A"] == 1.0
    assert result.loc["Model_A", "Model_B"] == result.loc["Model_B", "Model_A"]
