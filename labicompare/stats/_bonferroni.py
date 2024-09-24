import pandas as pd


def bonferroni_dunn(avg_ranks: pd.Series, n: int, alpha: float = 0.05) -> float:
    """
    Returns the critical difference for Bonferroni-Dunn according to a given alpha
    (either = 0.1 or 0.05) for the average ranking of N datasets.

    Args:
        avg_ranks (pd.Series): average ranking of the models
        n (int): number of datasets compared
        alpha (float): alpha level. must be either 0.05 or 0.1

    Returns:
        cd (float): critical difference value.

    Refs:
        [1] Dunn, O. J. (1961). Multiple Comparisons among Means. Journal of the American Statistical Association, 56(293), 52–64.
        [2] Demšar, Janez. "Statistical comparisons of classifiers over multiple data sets." The Journal of Machine learning research 7 (2006): 1-30.
    """
    ranks = avg_ranks.values
    k = ranks if isinstance(ranks, int) else len(ranks)

    bonferroni_critical_values = {
         0.05: [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576, 2.638, 2.690, 2.724, 2.773],
         0.1: [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326, 2.394, 2.450, 2.498, 2.539]
    }

    q = bonferroni_critical_values[alpha]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    
    return cd
