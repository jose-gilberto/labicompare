import pandas as pd
import numpy as np
from scipy.stats import studentized_range, norm
from abc import ABC, abstractmethod


def num_pairwise_comparisons(average_ranks: list[int]) -> float:
    k = len(average_ranks)
    m = (k * (k - 1)) // 2
    return m


class PostHocTest(ABC):
    """
    This class is an abstract class for implementing post hoc tests.
    It contains the basic methods and abstract signatures to facilitate implementations.
    """

    @abstractmethod
    def _critical_values(self) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def _correction(self) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def test(self, average_ranks: list[float], n_datasets: int, alpha: float = 0.05) -> float:
        raise NotImplementedError()


class Nemenyi(PostHocTest):
    """
    The Nemenyi test is a post-hoc test that compares pairs of classifiers to determine
    which differences are statistically significant.

    References:
        [1] Nemenyi, Peter Bjorn. Distribution-free multiple comparisons. Princeton University, 1963.
        [2] Demšar, Janez. "Statistical comparisons of classifiers over multiple data sets."
            The Journal of Machine learning research 7 (2006): 1-30.
    """
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _critical_values(k: int, n: int, alpha: float = 0.05) -> float:
        """ Computes the critical difference (CD) for the nemenyi test.

        Args:
            k (int): number of methods compared.
            n (int): number of datasets compared.
            alpha (float): alpha level.
        """
        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
        return cd

    @staticmethod
    def test(average_ranks: list[float], n: int, alpha: float = 0.05) -> float:
        """
        Returns the critical difference for Nemenyi post-hoc according to a given
        alpha for the average ranking of n datasets.

        Args:
            average_ranks (list[float]): average ranking of the models
            n (int): number of datasets compared
            alpha (float): alpha level.
        Returns:
            cd (float): critical difference value.
        """
        k = len(average_ranks)
        q_alpha = Nemenyi._critical_values(k, n, alpha)
        
        cd = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n))
        
        return cd


class BonferroniDunn(PostHocTest):
    """
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _critical_values(self, k: int, n: int, alpha: float = 0.05):
        """
        Calculates the critical difference (CD) for the Bonferroni-Dunn test.
        """
        z_alpha = norm.ppf(1 - alpha / (2 * (k - 1))) # Bonferroni correction
        cd = z_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
        return cd

    @staticmethod
    def test(average_ranks: list[float], n: int, alpha: float = 0.05) -> float:
        """
        Returns the critical difference for Bonferroni-Dunn post-hoc according to a given
        alpha for the average ranking of n datasets.

        CD is based on the Wilcoxon Signed-Rank test to compare pairs of classifiers.

        Args:
            average_ranks (list[float]): average ranking of the models
            n (int): number of datasets compared
            alpha (float): alpha level.
        Returns:
            cd (float): critical difference value.
        """
        k = len(average_ranks)
        cd = BonferroniDunn._critical_values(k, n, alpha)
        return cd
