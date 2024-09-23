import pandas as pd
import numpy as np
import operator
from typing import List, Tuple
from scipy.stats import friedmanchisquare, wilcoxon


def wilcoxon_holm(metrics: pd.DataFrame, alpha: float = 0.05) -> List[Tuple[str, str, float, bool]]:
    """"""
    friedman_p_value = friedmanchisquare(*(
        np.array(metrics[classifier].values) for classifier in metrics.columns
    ))[1]

    if friedman_p_value > alpha:
        raise ValueError('The null hypothesis over the entire classifiers could not be rejected.')

    m = metrics.columns.shape[0]
    p_values = []

    # Loop through the algorithms to perform a pair-wise test
    for i in range(m - 1):
        # Get the first classifier
        classifier_1 = metrics.columns[i]
        classifier_1_perf = metrics[classifier_1].values
        for j in range(i + 1, m):
            # Get the second classifier
            classifier_2 = metrics.columns[j]
            classifier_2_perf = metrics[classifier_2].values
            
            # Calculate the p-value
            p_value = wilcoxon(classifier_1_perf, classifier_2_perf, zero_method='pratt')[1] # Return stats, pvalue, zstats
            p_values.append((classifier_1, classifier_2, p_value, False))

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