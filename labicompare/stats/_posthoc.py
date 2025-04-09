import pandas as pd
import numpy as np
import itertools as it
import scipy.stats as ss

from abc import ABC, abstractmethod
from ._corrections import Correction



class PostHocTest(ABC):
    """
    This class is an abstract class for implementing post hoc tests.
    It contains the basic methods and abstract signatures to facilitate implementations.
    """

    @staticmethod
    @abstractmethod
    def test(metrics: pd.DataFrame, metric_name: str, correction: Correction = None) -> float:
        raise NotImplementedError()


class ConoverTest(PostHocTest):
    """ Executes the Conover post hoc test for multiple pairwise comparisons of 
    average ranks among groups (models), given a metric column from a long-format DataFrame.

    This method computes the differences in average ranks between all pairs of models,
    using the Conover-Iman test based on a t-statistic.

    Notes
    -----
    - Rankings are computed globally across all observations, without grouping by dataset.
    - The function applies a correction for tied ranks based on Conover's original method.
    - The test is two-sided and assumes homogeneity of group variances.
    - The significance of the p-values must be assessed with respect to a chosen alpha level.

    References
    ----------
    - Conover, W. J., & Iman, R. L. (1979). On multiple-comparisons procedures.
        Technical Report LA-7677-MS, Los Alamos Scientific Laboratory.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def test(metrics: pd.DataFrame,
             metric_name: str,
             correction: Correction = None) -> float:
        # Filter any nan value to avoid problems in calculus
        x = metrics[['model_name', 'dataset_name', metric_name]].dropna()
        x.columns = ['group', 'block', 'value']

        n = len(x)
        groups = x['group'].unique() # Get the name of all models used in comparison
        k = len(groups) # Number of all models
        x_lens = x.groupby('group')['value'].count()

        # Rank values within each dataset
        x['rank'] = x['value'].rank(method='average')
        x_ranks_average = x.groupby('group')['rank'].mean()
        x_ranks_sum = x.groupby('group')['rank'].sum()
        
        # Tie correction for ranking
        tie_sum = 0.
        for _, group_ in x.groupby('block'):
            counts = group_['value'].value_counts()
            ties = counts[counts > 1].values
            tie_sum += np.sum(ties ** 3 - ties)

        x_ties = 1. if tie_sum == 0 else 1. - tie_sum / (n ** 3 - n)

        # H-statistics corrected for Kruskall-Wallis
        h = (12. / (n * (n + 1.))) * np.sum(x_ranks_sum ** 2 / x_lens) - 3. * (n + 1.)
        h_cor = h / x_ties # Follow (Conover and Iman, 1979)

        # Compute the ranking variance
        if x_ties == 1:
            S2 = n * (n + 1.) / 12.
        else:
            S2 = (1.0 / (n - 1.)) * (np.sum(x['rank'] ** 2) - (n * (((n + 1.) ** 2.) / 4.)))

        # Init the comparison matrix for pair groups
        vs = np.zeros((k, k))
        group_labels = list(groups)
        combs = it.combinations(range(k), 2) # Pair-wise combinations

        def compare_conover(i, j):
            diff = np.abs(x_ranks_average.iloc[i] - x_ranks_average.iloc[j])
            B = 2. / x_lens.iloc[i]
            D = (n - 1 - h_cor) / (n - k) # Correction factor
            t_value = diff / np.sqrt(S2 * B * D)
            p_value = 2.0 * ss.t.sf(np.abs(t_value), df=n - k) # P-value
            return p_value
        
        for i, j in combs:
            vs[i, j] = compare_conover(i, j)
        
        if correction:
            tri_upper = np.triu_indices(k, 1)
            vs[tri_upper] = correction.apply(vs[tri_upper])[1] # Why? IDK

        vs += vs.T
        np.fill_diagonal(vs, 1)

        return pd.DataFrame(vs, index=group_labels, columns=group_labels)
