# Labicompare

## Installation

Labicompare requires at least a Python version of `3.9` or greater. Currently to install labicompare you can use the following command:

```sh
pip install git+https://github.com/jose-gilberto/labicompare/
```
Note that the package is currently on a alpha version. The commands used to install or the methods may change over time.

## Getting Started

As soon as you have the metrics for all models that you want to compare, we only need to put these values in a .csv file in the following format:

```
model_1,model_2,model_3
model_1_metric_1,model_2_metric_1,model_3_metric_1
model_1_metric_2,model_2_metric_2,model_3_metric_2
model_1_metric_3,model_2_metric_3,model_3_metric_3
```

Each column represents a model, in this case models 1, 2 and 3. Each row of the dataset represents a metric obtained on tha dataset of that row. For example, the first line may represent the metrics for dataset A obtained by the models 1, 2 and 3.

```python
import pandas as pd

metrics = pd.DataFrame({
    'CL1': [0.85, 0.88, 0.79],
    'CL2': [0.80, 0.90, 0.82],
    'CL3': [0.83, 0.87, 0.81]
}, index=['Dataset A', 'Dataset B', 'Dataset C'])
```

### Plotting 1v1 Comparison

To generate a one versus one plot comparison you only need to call the `one_vs_one_plot` function from the visualization module.

```python
from labicompare.visualization import one_vs_one_plot

one_vs_one_plot(
    metrics['CL1'].values,
    metrics['CL3'].values,
    'CL1', 'CL3'
)
```

This code will generate the following figure:

![One versus One Example](./docs/assets/one_vs_one_example.png)

### Plotting a Critical Difference Diagram

## How to Contribute

## Issues

## Acknowledgements and References

## Next Steps/TODO
