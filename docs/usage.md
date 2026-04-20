
# How to Use?

## Data Handling

The `EvaluationData` class is the main gateway into **labicompare**. Before you can run any statistical tests or generate plost, you must wrap your raw results in this class. It handles data validation, missing value treatment, and automatic ranking generation.

### 1. Preparing Your Data

`labicompare` expects your data to be a standard `pandas.DataFrame` structured as a matrix:
- **Columns** represent the **Models** you are comparing.
- **Rows** represent the **Datasets**, Cross-Validation Folds, or independent experimental runs.

**Input:**

```python
import pandas as pd
import numpy as np

data_dict = {
  'RandomForest': [0.85, 0.88, np.nan, 0.89], # Note the missing value
  'XGBoost': [0.86, 0.89, 0.84, 0.90],
  'SVM': [0.70, 0.75, 0.72, 0.71]
}

df = pd.DataFrame(
  data_dict,
  index=['Dataset_1', 'Dataset_2', 'Dataset_3', 'Dataset_4']
)

print(df)
```

**Outpus:**

```text
          RandomForest   XGBoost   SVM
Dataset_1          0.85     0.86  0.70
Dataset_2          0.88     0.89  0.75
Dataset_3           NaN     0.84  0.72
Dataset_4          0.89     0.90  0.71
```

### 2. Instantiating EvaluationData

When creating the object, you must define the `higher_is_better` parameter.

- Use `True` for metrics like Accuracy, F1-Score, and AUC (where bigger is better, so the highest value gets Rank 1).
- Use `False` for error metrics like RMSE or MAE (where smaller is better, so the lowest value gets Rank 1).

**Input:**

```python
from labicompare.core.data import EvaluationData

# Assuming Accuracy (higher is better)
eval_data = EvaluationData(data=df, higher_is_better=True)
print(eval_data)
```

**Output:**

```text
WARNING: Null values detected. Rows (or datasets) with NaNs will be removed to ensure the integrity of paired statistical tests and methods.
<EvaluationData: 3 datasets, 3 models>
```

**Important Note on NaNs**: Statistical tests for multiple comparisons (like Friedman) are paired. If a dataset is missing a score for even one model, `EvaluationData` automatically drops the entire row (dataset) to maintain the integrity of the tests. Notice that Dataset_3 was dropped, leaving 3 datasets.

### 3. Manipulating and Accessing Attributes

Once initialized, EvaluationData computes the ranks automatically and exposes several useful attributes. You do not need to compute averages manually.

#### Accessing Model and Dataset Names

**Input:**

```python
print("Models:", eval_data.model_names)
print("Datasets:", eval_data.dataset_names)
```

**Output:**

```plaintext
Models: ['RandomForest', 'XGBoost', 'SVM']
Datasets: ['Dataset_1', 'Dataset_2', 'Dataset_4']
```

#### Accessing the Ranks Matrix

The class automatically calculates the ranks for each dataset. If there are ties, it assigns the average rank.

**Input:**

```python
# Access the pandas DataFrame of ranks
print(eval_data.ranks_df)
```

**Output:**

```text
            RandomForest  XGBoost  SVM
Dataset_1           2.0      1.0  3.0
Dataset_2           2.0      1.0  3.0
Dataset_4           2.0      1.0  3.0
```

#### Accessing Raw NumPy Arrays

If you need to build custom plots or pass the data to external mathematical functions, you can extract the raw NumPy matrices:

**Input:**

```python
# Raw scores (without NaNs)
print(eval_data.scores)

# Raw ranks
print(eval_data.ranks)
```

**Output:**

```text
[[0.85 0.86 0.7 ]
 [0.88 0.89 0.75]
 [0.89 0.9  0.71]]

[[2. 1. 3.]
 [2. 1. 3.]
 [2. 1. 3.]]
```

## Statistical Tests & Comparison Summary

The `labicompare.stats` module provides the statistical engine to validate whether the differences in model performances are real or just due to random chance.

We follow the standard non-parametric procedure recommended for Machine Learning benchmarks:  

1. **Global Test (Friedman Test):** Checks if there is *any* significant difference among the models overall.  
2. **Post-hoc Test (Wilcoxon Signed-Rank with Holm's step-down procedure):** If the global test finds a difference, this test compares the models pairwise to find out *which specific models* differ from each other, adjusting the p-values to prevent false positives (controlling the Family-Wise Error Rate). 

The labicompare.stats module is divided into specialized components to handle each stage of the statistical validation pipeline. This modular approach allows for fine-grained control over the evaluation process.

### Statistical Pipeline

To ensure scientific validity, the comparison follows a three-step sequence:  

1. **Global Test:** Run the Friedman test to check for overall differences.  

2. **Pairwise Comparison:** Compute raw p-values for every model pair.  

3. **Post-hoc Adjustment:** Apply the Wilcoxon-Holm procedure to correct p-values for multiple comparisons.  


#### Global Test (`friedman` module)

The Friedman test evaluates if at least one model performs significantly differently from the others.

```python
from labicompare.stats.friedman import friedman_test

# Returns the statistic and the global p-value
statistic, p_value = friedman_test(eval_data)
print(f"Friedman P-Value: {p_value:.5f}")
```

#### Post-hoc Tests (`posthoc` module)

The `labicompare.stats.posthoc` module contains the `wilcoxon_holm` function. This function is the core statistical engine of the library, acting as an all-in-one pipeline that validates your data from the global level down to individual pairwise comparisons.

##### How It Works Under the Hood

When you call `wilcoxon_holm`, it automatically executes a rigorous three-step procedure:

1.  **Global Significance Check (Friedman Test):** It first runs the Friedman test. If the p-value is greater than your `alpha`, it raises a `ValueError` and stops. This strict behavior prevents you from reporting "false positive" pairwise differences when the models are globally equivalent.
2.  **Pairwise Wilcoxon Signed-Rank Test:** For every possible pair of models, it computes the raw p-value and calculates the mean difference to determine the "winner" (respecting your `higher_is_better` flag).
3.  **Holm's Step-Down Correction:** It sorts the raw p-values from smallest to largest and adjusts the significance threshold iteratively using the formula $p_i \leq \frac{\alpha}{k - i}$ (where $k$ is the total number of comparisons).

##### 1. Standard Usage

**Input:**
```python
from labicompare.stats.posthoc import wilcoxon_holm

# Assuming eval_data is your instantiated EvaluationData object
# The default alpha is 0.05
summary = wilcoxon_holm(data=eval_data, alpha=0.05)

print(summary)
```
**Output:**
```text
<ComparisonSummary: Friedman P-value=0.0012, 3 Pairwise Comparisons>
```

### Unpacking the `ComparisonSummary`

The function returns a rich `ComparisonSummary` object containing everything you need for plotting and reporting.

**Input:**

```python
# 1. Global Test Results
print(f"Friedman Stat: {summary.friedman_stat:.2f}")
print(f"Global P-Value: {summary.friedman_p_value:.5f}\n")

# 2. Pairwise Results
for res in summary.pairwise_results:
    print(f"{res.model_a} vs {res.model_b}:")
    print(f"  Winner: {res.winner}")
    print(f"  P-value: {res.p_value:.5f}")
    print(f"  Significant: {res.is_significant}\n")
```

**Output:**
```text
Friedman Stat: 12.50
Global P-Value: 0.00193

XGBoost vs SVM:
  Winner: XGBoost
  P-value: 0.00312
  Significant: True

RandomForest vs SVM:
  Winner: RandomForest
  P-value: 0.04100
  Significant: True

XGBoost vs RandomForest:
  Winner: XGBoost
  P-value: 0.15400
  Significant: False
```
