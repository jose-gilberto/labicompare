import warnings

import numpy as np
import scipy.stats as st

from labicompare.core.data import EvaluationData
from labicompare.core.results import PairwiseResult


def _determine_winner(
  mean_diff: float,
  model_a: str,
  model_b: str,
  higher_is_better: bool
) -> str | None:
  """ Method to determine the winner based on mean differences."""
  if mean_diff == 0:
    return None
  if higher_is_better:
    return model_a if mean_diff > 0 else model_b
  else:
    return model_a if mean_diff < 0 else model_b


def paired_ttest(
  data: EvaluationData, 
  model_a: str, 
  model_b: str, 
  alpha: float = 0.05,
  check_normality: bool = True
) -> PairwiseResult:
  """
  Runs the Paired T-Test (Parametric) between two models.
  """
  if model_a not in data.model_names or model_b not in data.model_names:
    raise ValueError("One or both models not found in data.")

  perf_a = data._df[model_a].values
  perf_b = data._df[model_b].values

  diffs = perf_a - perf_b

  if check_normality and len(diffs) >= 3:
    shapiro_stat, shapiro_p = st.shapiro(diffs)
    
    if shapiro_p < alpha:
      warnings.warn(
        f"\n[labicompare] WARNING:\n"
        f"Differences between '{model_a}' and '{model_b}' NOT follow a normal "
        f"distribution (Shapiro-Wilk p-value = {shapiro_p:.4f} < {alpha}).\n"
        f"The result of this paired T-Test  has high risk of false positive. "
        f"We strongly suggest using the Wilcoxon Signed-Rank instead.",
        UserWarning,
        stacklevel=2
      )

  res = st.ttest_rel(perf_a, perf_b)
  p_value = float(res.pvalue)

  mean_diff = float(np.mean(diffs))
  is_significant = p_value <= alpha
  winner = _determine_winner(mean_diff, model_a, model_b, data.higher_is_better)

  return PairwiseResult(
    model_a=model_a,
    model_b=model_b,
    p_value=p_value,
    is_significant=is_significant,
    winner=winner if is_significant else None,
    mean_diff=mean_diff
  )


def sign_test(
  data: EvaluationData, 
  model_a: str, 
  model_b: str, 
  alpha: float = 0.05
) -> PairwiseResult:
  """
  Runs the sign-rank test (non-parametric).
  Based in only who wins or loose each round, ignoring the scale of each differences.
  """
  if model_a not in data.model_names or model_b not in data.model_names:
    raise ValueError("One or both models not found in results.")

  perf_a = data._df[model_a].values
  perf_b = data._df[model_b].values

  diffs = perf_a - perf_b
  non_zero_diffs = diffs[diffs != 0]
  n_trials = len(non_zero_diffs)

  mean_diff = float(np.mean(diffs))

  if n_trials == 0:
    p_value = 1.0
  else:
    positive_signs = np.sum(non_zero_diffs > 0)

    res = st.binomtest(k=positive_signs, n=n_trials, p=0.5, alternative='two-sided')
    p_value = float(res.pvalue)

  is_significant = p_value <= alpha
  winner = _determine_winner(mean_diff, model_a, model_b, data.higher_is_better)

  return PairwiseResult(
    model_a=model_a,
    model_b=model_b,
    p_value=p_value,
    is_significant=is_significant,
    winner=winner if is_significant else None,
    mean_diff=mean_diff
  )


def wilcoxon_signed_rank(
  data: EvaluationData, 
  model_a: str, 
  model_b: str, 
  alpha: float = 0.05
) -> PairwiseResult:
  """
  Executes the Wilcoxon signed-rank test (non-parametric) between two models.
  This is the ideal alternative for paired T-Test when the data do not follow
  a normal distribution. Consider the direction (who wins) and the scale of ranking
  differences.
  """
  if model_a not in data.model_names or model_b not in data.model_names:
    raise ValueError("One or both models not found in results.")

  perf_a = data._df[model_a].values
  perf_b = data._df[model_b].values

  try:
      res = st.wilcoxon(perf_a, perf_b, zero_method='pratt')
      p_value = float(res.pvalue)
  except ValueError as e:
      if "zero_method" in str(e) or "zero" in str(e):
          p_value = 1.0
      else:
          raise e
  
  mean_diff = float(np.mean(perf_a - perf_b))
  is_significant = p_value <= alpha
  winner = _determine_winner(mean_diff, model_a, model_b, data.higher_is_better)

  return PairwiseResult(
    model_a=model_a,
    model_b=model_b,
    p_value=p_value,
    is_significant=is_significant,
    winner=winner if is_significant else None,
    mean_diff=mean_diff
  )
