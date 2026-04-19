import scipy.stats as st

from labicompare.core.data import EvaluationData


def friedman_test(data: EvaluationData) -> tuple[float, float]:
  """
  Apply the Friedman test with Iman-Davenport correction [1].
  
  Args:
    data: EvaluateData instance containing the calculated ranks.
    
  Returns:
    An tuple containing (chi-squared stats, p-value)
  
  References:
  .. [1] Demšar, Janez. "Statistical comparisons of classifiers over
      multiple data sets." Journal of Machine learning research
      7.Jan (2006): 1-30.
  """
  if len(data.model_names) < 3:
    raise ValueError(
      "Friedman test requires at least 3 models for comparison."
      "For 2 models, you should use Wilcoxon test."
    )

  model_arrays = [data._df[model].values for model in data.model_names]
  res = st.friedmanchisquare(*model_arrays)

  return float(res.statistic), float(res.pvalue)
