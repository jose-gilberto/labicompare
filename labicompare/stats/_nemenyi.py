import pandas as pd


def nemenyi(avg_ranks: pd.Series, n: int, alpha: float = 0.05) -> float:
    """
    Returns the critical difference for Nemenyi according to a given alpha
    (either = 0.1 or 0.05) for the average ranking of N datasets.

    Args:
        avg_ranks (pd.Series): average ranking of the models
        n (int): number of datasets compared
        alpha (float): alpha level. must be either 0.05 or 0.1

    Returns:
        cd (float): critical difference value.

    Refs:
        [1] Nemenyi, Peter Bjorn. Distribution-free multiple comparisons. Princeton University, 1963.
        [2] Demšar, Janez. "Statistical comparisons of classifiers over multiple data sets." The Journal of Machine learning research 7 (2006): 1-30.
    """
    ranks = avg_ranks.values
    k = ranks if isinstance(ranks, int) else len(ranks)

    nemenyi_critical_values = {
        0.05: [0, 0, 1.959964, 2.343701, 2.569032, 2.727774, 2.849705, 2.94832,
               3.030879, 3.101730, 3.163684, 3.218654, 3.268004, 3.312739, 3.353618,
               3.39123, 3.426041, 3.458425, 3.488685, 3.517073, 3.543799],
        0.1:  [0, 0, 1.644854, 2.052293, 2.291341, 2.459516, 2.588521, 2.692732,
               2.779884, 2.854606, 2.919889, 2.977768, 3.029694, 3.076733, 3.119693,
               3.159199, 3.195743, 3.229723, 3.261461, 3.291224, 3.319233]
    }

    q = nemenyi_critical_values[alpha]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5

    return cd
