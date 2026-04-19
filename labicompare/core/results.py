from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PairwiseResult:
  model_a: str
  model_b: str
  p_value: float
  is_significant: bool
  winner: Optional[str]
  mean_diff: float
  
@dataclass
class ComparisonSummary:
  friedman_stat: float
  friedman_p_value: float
  is_global_sig: bool
  
  pairwise_results: list[PairwiseResult]
  
  model_means: dict[str, float]

  alpha: float
  higher_is_better: bool
  n_samples: int
  
  def to_dataframe(self) -> pd.DataFrame:
    data = []
    for res in self.pairwise_results:
      data.append({
        "Model A": res.model_a,
        "Model B": res.model_b,
        "P-Value": res.p_value,
        "Significant": res.is_significant,
        "Winner": res.winner if res.winner else "Tie",
        "Mean Diff": res.mean_diff
      })
    return pd.DataFrame(data)
  
  def get_leaderboard(self) -> pd.DataFrame:
    df = pd.Series(self.model_means).to_frame("Mean_Performance")
    df = df.sort_values(by="Mean_Performance", ascending=not self.higher_is_better)
    df["Rank"] = range(1, len(df) + 1)
    return df

  def __repr__(self) -> str:
    sig_str = "REJECTED" if self.is_global_sig else "NOT REJECTED"
    return (
      f"ComparisonSummary(Friedman P-Value={self.friedman_p_value:.4f}, "
      f"H0={sig_str}, Models={len(self.model_means)})"
    )