import pandas as pd
import numpy as np
import operator
from typing import List, Tuple
from scipy.stats import friedmanchisquare, wilcoxon


def wilcoxon_holm(
    metrics: pd.DataFrame,
    alpha: float = 0.05
) -> Tuple[List[Tuple[str, str, float, bool]], int]:
    """
    Returns the p_values for each pairwise comparison execute between the models provided
    in the dataset for Wilcoxon pairwise test with Holm alpha correction for a given alpha value.
    
    The method uses a Friedman test to reject the null hypothesis of the entire models being
    statistically equivalent.

    Args:
        metrics (pd.DataFrame): dataframe with the metrics for each model over the entire datasets.
        alpha (float): alpha level.

    Returns:
        p_values (List[Tuple[str, str, float, bool]]): list of tuples containing each result for the pairwise test
            between model AxB. Each tuple contain the name of the two compared methods in the first two positions
            the p_value on the third position and a bool value indicating if the methods are statistically different
            in the last position.
        m (int): number of datasets used in the comparison.

    Refs:
        [1] M. Friedman, “A comparison of alternative tests of significance for the problem of m rankings,” The Annals of Mathematical Statistics, vol. 11, no. 1, pp. 86–92, 1940.
        [2] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945.
        [3] S. Holm, “A simple sequentially rejective multiple test procedure, ”Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.
        [4] Demšar, Janez. "Statistical comparisons of classifiers over multiple data sets." The Journal of Machine learning research 7 (2006): 1-30.
    """
    friedman_p_value = friedmanchisquare(*(
        np.array(metrics[model].values) for model in metrics.columns
    ))[1]

    if friedman_p_value > alpha:
        raise ValueError('The null hypothesis over the entire models could not be rejected.')

    m = metrics.columns.shape[0]
    p_values = []

    # Loop through the algorithms to perform a pair-wise test
    for i in range(m - 1):
        # Get the first model
        model_1 = metrics.columns[i]
        model_1_perf = metrics[model_1].values

        for j in range(i + 1, m):
            # Get the second model
            model_2 = metrics.columns[j]
            model_2_perf = metrics[model_2].values

            # Calculate the p-value
            p_value = wilcoxon(model_1_perf, model_2_perf, zero_method='pratt')[1] # Return stats, pvalue, zstats
            p_values.append((model_1, model_2, p_value, False))

    k = len(p_values) # Get the number of hypothesis
    p_values.sort(key=operator.itemgetter(2))

    # Loop through the hypothesis
    for i in range(k):
        new_alpha = float(alpha / (k - i))
        # Test if significant after holm's correction alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            break

    return p_values, m