import itertools

import numpy as np
import pandas as pd

from typing import Any

from labicompare.core.data import EvaluationData
from labicompare.stats.wilcoxon import wilcoxon_test


def adjust_holm(p_values: list[float]) -> list[float] | Any:
    """
    Applies top-down Holm-Bonferroni correction to control the error
    (FWER) in multiple tests.
    """
    n = len(p_values)
    if n == 0:
        return []
    
    p_array = np.array(p_values)
    sorted_indices = np.argsort(p_values)

    adjusted_p = np.zeros(n)
    running_max = 0.0

    for step, idx in enumerate(sorted_indices):
        multiplier = n - step

        adj = min(p_array[idx] * multiplier, 1.0)

        running_max = max(running_max, adj)
        adjusted_p[idx] = running_max

    return adjusted_p.tolist()


def posthoc_wilcoxon(data: EvaluationData) -> pd.DataFrame:
    """
    Run pair-wise comparisons between all models using wilcoxon test and applies
    Holm-Bonferroni correction.

    Args:
        data: EvaluationData instance containing all results.
    
    Returns:
        pandas.DataFrame symetric containing all adjusted p-values.
    """
    models = data.model_names
    n_models = len(models)

    results = pd.DataFrame(
        np.ones((n_models, n_models)), index=models, columns=models
    )

    pairs = list(itertools.combinations(models, 2))

    if not pairs:
        return results
    
    p_values = []
    for m1, m2 in pairs:
        _, p = wilcoxon_test(data, model_1=m1, model_2=m2)
        p_values.append(p)

    adjusted_p = adjust_holm(p_values)

    for (m1, m2), p_adj in zip(pairs, adjusted_p):
        results.loc[m1, m2] = p_adj
        results.loc[m2, m1] = p_adj

    return results