import pandas as pd


def bonferroni_dunn(avg_ranks: pd.Series, n: int, alpha: float = 0.05) -> float:
    """ TODO: write docs for that function
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
