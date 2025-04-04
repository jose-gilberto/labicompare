from abc import ABC, abstractmethod
import numpy as np


class Correction(ABC):

    @staticmethod
    @abstractmethod
    def apply(p_values: list[float],
              alpha: float = 0.05,
              max_iter: int = 1,
              is_sorted: bool = False,
              return_sorted: bool = False) -> tuple[bool, list[float], float, float]:
        pass


class BonferroniCorrection(Correction):
    """
    Applies the Bonferroni correction to a list of p-values to control for the 
    family-wise error rate (FWER) in multiple comparisons.

    The Bonferroni correction is a conservative adjustment method that reduces 
    the likelihood of Type I errors (false positives) when conducting multiple 
    statistical tests. It adjusts the significance threshold by dividing alpha 
    by the number of comparisons (m), ensuring that the probability of making 
    at least one false rejection of the null hypothesis remains at most alpha.

    Notes
    -----
    - The Bonferroni correction is **highly conservative**, especially when the 
      number of comparisons is large, which may increase the risk of Type II 
      errors (false negatives).
    - For a less strict correction, alternatives like the Holm or Benjamini-Hochberg 
      methods can be considered.

    References
    ----------
    - Bonferroni, C. (1936). "Teoria statistica delle classi e calcolo delle probabilità". 
      Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze.
    - Holm, S. (1979). "A simple sequentially rejective multiple test procedure". 
      Scandinavian Journal of Statistics, 6(2), 65-70.
    - Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate: 
      A practical and powerful approach to multiple testing". Journal of the 
      Royal Statistical Society: Series B (Methodological), 57(1), 289-300.
    """
    
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def apply(p_values,
              alpha = 0.05,
              max_iter = 1,
              is_sorted = False,
              return_sorted = False) -> tuple[list[bool], list[float], float, float]:
        p_values = np.asarray(p_values)
        alpha_f = alpha

        if not is_sorted:
            sort_idxs = np.argsort(p_values)
            p_values = np.take(p_values, sort_idxs)

        n_tests = len(p_values)
        alpha_sidak = 1 - np.power((1. - alpha_f), (1. / n_tests))
        alpha_bonferroni = alpha_f / float(n_tests)

        reject = p_values <= alpha_bonferroni
        p_values_corrected = p_values * float(n_tests)

        if p_values_corrected is not None:
            p_values_corrected[p_values_corrected > 1] = 1

        if is_sorted or return_sorted:
            return reject, p_values_corrected, alpha_sidak, alpha_bonferroni
        else:
            p_values_corrected_ = np.empty_like(p_values_corrected)
            p_values_corrected_[sort_idxs] = p_values_corrected
            del p_values_corrected

            reject_ = np.empty_like(reject)
            reject_[sort_idxs] = reject
            return reject_, p_values_corrected_, alpha_sidak, alpha_bonferroni


class SidakCorrection(Correction):
    """
    Applies the Sidak correction to a list of p-values to control the family-wise 
    error rate (FWER) in multiple hypothesis testing.

    The Šidák correction is a statistical adjustment method that, unlike Bonferroni, 
    assumes that multiple tests are **independent**. It is slightly less conservative 
    than Bonferroni and adjusts the significance threshold using the formula:
        
        alpha_corrected = 1 - (1 - alpha)^(1/m)

    where `m` is the number of comparisons.

    The corrected p-values are computed as:
    
        p_values_corrected = 1 - (1 - p)^n

    This correction ensures that the overall probability of making at least one Type I 
    error does not exceed `alpha`.

    Notes
    -----
    - The Sidak correction assumes **independence** of tests. If tests are 
      positively correlated, it behaves similarly to the Bonferroni correction.
    - It is **less conservative than Bonferroni**, leading to fewer false negatives 
      (Type II errors).
    - If the number of comparisons is large, other methods like Holm-Sidak or 
      Benjamini-Hochberg may be preferable.

    References
    ----------
    - Šidák, Z. (1967). "Rectangular Confidence Regions for the Means of 
      Multivariate Normal Distributions". Journal of the American Statistical 
      Association, 62(318), 626-633.
    - Holm, S. (1979). "A simple sequentially rejective multiple test procedure". 
      Scandinavian Journal of Statistics, 6(2), 65-70.
    - Abdi, H. (2007). "Bonferroni and Sidak corrections for multiple comparisons". 
      In *Encyclopedia of Measurement and Statistics* (Vol. 3, pp. 103-107).
    """
    
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def apply(p_values, alpha = 0.05, max_iter = 1, is_sorted = False, return_sorted = False):
        p_values = np.asarray(p_values)
        alpha_f = alpha

        if not is_sorted:
            sort_idxs = np.argsort(p_values)
            p_values = np.take(p_values, sort_idxs)

        n_tests = len(p_values)
        alpha_sidak = 1 - np.power((1. - alpha_f), (1. / n_tests))
        alpha_bonferroni = alpha_f / float(n_tests)

        reject = p_values <= alpha_sidak
        p_values_corrected = np.expm1(n_tests * np.log1p(-p_values))

        if p_values_corrected is not None:
            p_values_corrected[p_values_corrected > 1] = 1

        if is_sorted or return_sorted:
            return reject, p_values_corrected, alpha_sidak, alpha_bonferroni
        else:
            p_values_corrected_ = np.empty_like(p_values_corrected)
            p_values_corrected_[sort_idxs] = p_values_corrected
            del p_values_corrected

            reject_ = np.empty_like(reject)
            reject_[sort_idxs] = reject
            return reject_, p_values_corrected_, alpha_sidak, alpha_bonferroni


class HolmSidakCorrection(Correction):
    """
    Applies the Holm-Sidak correction for multiple hypothesis testing, controlling 
    the Family-Wise Error Rate (FWER) while being less conservative than the 
    Bonferroni correction.

    The Holm-Sidak method is a **step-down procedure** that adjusts the p-values in 
    an ordered sequence. It controls the probability of making at least one Type I 
    error when performing multiple comparisons.

    The correction follows these steps:

    1. **Sort** the p-values in ascending order.
    2. **Compute adjusted significance thresholds** for each hypothesis:

        alpha_corrected(i) = 1 - (1 - alpha)^(1 / (m - i + 1))

    where:
       - `i` is the rank (starting from the smallest p-value)
       - `m` is the total number of hypotheses

    3. **Compare each p-value with its adjusted threshold**. The first **non-significant** 
       p-value (p > alpha_corrected(i)) means all subsequent hypotheses are also considered 
       non-significant.

    The Holm-Sidak method is more **powerful than Bonferroni** because it applies 
    progressively larger significance thresholds as more hypotheses are rejected.

    Notes
    -----
    - The Holm-Sidak method is **less conservative than Bonferroni**, reducing Type II errors.
    - It assumes **independent or weakly correlated** tests.
    - If tests are strongly correlated, adjustments may still be too strict.
    - Holm-Sidak is a stepwise improvement over Sidak and Holm-Bonferroni.


    References
    ----------
    - Holm, S. (1979). "A simple sequentially rejective multiple test procedure". 
      Scandinavian Journal of Statistics, 6(2), 65-70.
    - Sidak, Z. (1967). "Rectangular Confidence Regions for the Means of Multivariate 
      Normal Distributions". Journal of the American Statistical Association, 62(318), 626-633.
    - Abdi, H. (2010). "Holm's sequential Bonferroni procedure". In *Encyclopedia of 
      Research Design* (Vol. 1, pp. 1-8).
    """
    
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def apply(p_values, alpha = 0.05, max_iter = 1, is_sorted = False, return_sorted = False):
        p_values = np.asarray(p_values)
        alpha_f = alpha

        if not is_sorted:
            sort_idxs = np.argsort(p_values)
            p_values = np.take(p_values, sort_idxs)

        n_tests = len(p_values)
        alpha_sidak = 1 - np.power((1. - alpha_f), (1. / n_tests))
        alpha_bonferroni = alpha_f / float(n_tests)

        alpha_sidak_all = 1 - np.power(1 - np.power((1. - alpha_f), (1. / np.arange(n_tests, 0, -1))))
        not_reject = p_values > alpha_sidak_all
        del alpha_sidak_all

        nr_index = np.nonzero(not_reject)[0]
        if nr_index.size == 0:
            not_reject_min = len(p_values)
        else:
            not_reject_min = np.min(nr_index)

        not_reject[not_reject_min:] = True
        reject = ~not_reject
        del not_reject

        p_values_corrected_raw = -np.expm1(np.arange(n_tests, 0, -1) * np.log1p(-p_values))
        p_values_corrected = np.maximum.accumulate(p_values_corrected_raw)
        del p_values_corrected_raw

        if p_values_corrected is not None:
            p_values_corrected[p_values_corrected > 1] = 1

        if is_sorted or return_sorted:
            return reject, p_values_corrected, alpha_sidak, alpha_bonferroni
        else:
            p_values_corrected_ = np.empty_like(p_values_corrected)
            p_values_corrected_[sort_idxs] = p_values_corrected
            del p_values_corrected

            reject_ = np.empty_like(reject)
            reject_[sort_idxs] = reject
            return reject_, p_values_corrected_, alpha_sidak, alpha_bonferroni

