import numpy as np
import scipy.stats as st

from labicompare.core.data import EvaluationData


def friedman_test(data: EvaluationData) -> tuple[float, float]:
  """
  Apply the Friedman test with Iman-Davenport correction [1].
  
  Args:
    data: EvaluateData instance containing the calculated ranks.
    
  Returns:
    An tuple containing (F statistics, p-value)
  
  [1] Demšar, Janez. "Statistical comparisons of classifiers over
      multiple data sets." Journal of Machine learning research
      7.Jan (2006): 1-30.
  """
  n_datasets, n_models = data.ranks.shape
  
  if n_models < 3:
    raise ValueError(
      "Friedman test requires at least 3 models for comparison."
      "For 2 models, you should use Wilcoxon test."
    )
    
  mean_ranks = np.mean(data.ranks, axis=0)
  chi2_f = (12 * n_datasets / (n_models * (n_models + 1))) * (
    np.sum(mean_ranks ** 2) - (n_models * (n_models + 1) ** 2) / 4
  )
  
  # Iman-Davenport correction
  denominator = (n_datasets * (n_models - 1) - chi2_f)
  if denominator == 0:
    f_stat = 0.0
    p_value = 1.0
  else:
    f_stat = (n_datasets - 1) * chi2_f / denominator
    
    df1 = n_models - 1
    df2 = (n_models - 1) * (n_datasets - 1)
    
    p_value = st.f.sf(f_stat, df1, df2)

  return float(f_stat), float(p_value)
