import numpy as np
import pandas as pd
import pytest

from labicompare.core.data import EvaluationData


def test_evaluation_data_initialization() -> None:
  """Test if the EvaluationData class loads correctly a clean dataset."""
  df = pd.DataFrame(
    {"Model_A": [0.8, 0.85], "Model_B": [0.75, 0.9]},
    index=["Dataset_1", "Dataset_2"]
  )
  
  eval_data = EvaluationData(df)
  
  assert eval_data.model_names == ["Model_A", "Model_B"]
  assert eval_data.dataset_names == ["Dataset_1", "Dataset_2"]
  assert eval_data.scores.shape == (2, 2)
  

def test_evaluation_data_remove_nans() -> None:
  """Test if the EvaluationData class removes rows with NaNs correctly."""
  df = pd.DataFrame(
    {"Model_A": [0.8, np.nan, 0.9], "Model_B": [0.75, 0.8, 0.85]},
    index=["Dataset_1", "Dataset_2", "Dataset_3"]
  )
  
  eval_data = EvaluationData(df)
  
  assert len(eval_data.dataset_names) == 2
  assert "Dataset_2" not in eval_data.dataset_names
  
  
def test_evaluation_data_type_error() -> None:
  """Test if the EvaluationData class triggers an error if receives not an DataFrame."""
  with pytest.raises(TypeError):
    EvaluationData([0.9, 0.8])


def test_evaluation_data_ranking_higher_is_better() -> None:
  """Test if the EvaluationData class calculate ranks correctly maximizing scores."""
  # dataset 1 -> A is better, B and C ties
  # dataset 2 -> C is better, following B and A
  df = pd.DataFrame(
    {
      "Model_A": [0.9, 0.7],
      "Model_B": [0.8, 0.8],
      "Model_C": [0.8, 0.9],
    },
    index=["Dataset_1", "Dataset_2"]
  )
  
  eval_data = EvaluationData(df)
  
  np.testing.assert_array_equal(eval_data.ranks[0], [1.0, 2.5, 2.5])
  np.testing.assert_array_equal(eval_data.ranks[1], [3.0, 2.0, 1.0])


def test_evaluation_data_ranking_lower_is_better() -> None:
  """"Test if the EvaluationData class calculate ranks correctly minimizing scores."""
  df = pd.DataFrame(
    {
      "Model_A": [10.5, 5.0], 
      "Model_B": [12.0, 15.0]
    }
  )
  
  eval_data = EvaluationData(df, higher_is_better=False)
  
  np.testing.assert_array_equal(eval_data.ranks[0], [1.0, 2.0])
  np.testing.assert_array_equal(eval_data.ranks[1], [1.0, 2.0])
