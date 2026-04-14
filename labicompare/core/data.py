import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationData:
  """
  """
  
  def __init__(self, data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Input 'data' must be a pandas.DataFrame.")

    self._df = data.copy()
    
    if self._df.isnull().values.any():
      logger.warning(
        "Null values detected. Rows (or datasets) with NaNs will be removed "
        "to ensure the integrity of paired statistical tests and methods."
      )
      self._df = self._df.dropna()
      
    if self._df.empty:
      raise ValueError(
        "The input DataFrame is empty (or at least become empty after removing NaNs)."
      )
      
    self.model_names: list[str] = self._df.columns.tolist()
    self.dataset_names: list[str] = self._df.index.tolist()
    
    self.scores: np.ndarray = self._df.values
    
  def __repr__(self) -> str:
    return (
      f"<EvaluationData: {len(self.dataset_names)} datasets, "
      f"{len(self.model_names)} models>"
    )