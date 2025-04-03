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
