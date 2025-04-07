import pandas as pd
import numpy as np
from scipy.stats import studentized_range, norm
from abc import ABC, abstractmethod
from ._corrections import Correction


def num_pairwise_comparisons(average_ranks: list[int]) -> float:
    k = len(average_ranks)
    m = (k * (k - 1)) // 2
    return m


class PostHocTest(ABC):
    """
    This class is an abstract class for implementing post hoc tests.
    It contains the basic methods and abstract signatures to facilitate implementations.
    """
    
    @staticmethod
    @abstractmethod
    def test(self, metrics: pd.DataFrame, alpha: float = 0.05, correction: Correction = None) -> float:
        raise NotImplementedError()


class Conover(PostHocTest):

    def __init__(self):
        super().__init__()

    @staticmethod
    def compare_conover(i, j):
        ...

    @staticmethod
    def test(self,
             metrics: pd.DataFrame,
             metric_name: str,
             alpha: float = 0.05,
             correction: Correction = None) -> float:
        metrics = metrics.sort_values(by=['model', 'dataset'], ascending=True)
        n = metrics['dataset'].unique().shape[0]
        metrics['rank'] = metrics.groupby('dataset')[metric_name].rank(ascending=False)
        
        metrics_ranks_avg = metrics.groupby('model', observed=True)["rank"].mean()
        metrics_ranks_sum = metrics.groupby('model', observed=True)["rank"].sum().to_numpy()

        values = metrics.groupby('rank').count()[metric_name].to_numpy()

        tie_sum = np.sum(values[values != 1] ** 3 - values[values != 1])
        tie_sum = 0 if not tie_sum else tie_sum

        metrics_ties = np.min([1.0, 1.0 - tie_sum / (n ** 3 - n)])

        h = (12. / (n * (n + 1.))) * np.sum(metrics_ranks_sum ** 2 / n) - 3. * (n + 1.)
        h_cor = h / metrics_ties

        if metrics_ties == 1:
            S2 = n * (n + 1) / 12
        else:
            S2 = (1. / (n - 1.)) * (np.sum(metrics['rank'] ** 2) - ())
