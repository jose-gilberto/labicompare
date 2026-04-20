import numpy as np
import pandas as pd

from labicompare.core.data import EvaluationData
from labicompare.plots.differences import plot_difference_distribution
from labicompare.stats.pairwise import paired_ttest

df = pd.read_csv('./results.csv', index_col='dataset')
data = EvaluationData(df, higher_is_better=True)

res_t = paired_ttest(data, "InceptionTime", "FCN")
print(f"T-Test P-Value: {res_t.p_value:.4f}")

fig = plot_difference_distribution(data, "InceptionTime", "FCN", figsize=(9, 4))
fig.savefig("differences_distributions.png", dpi=300, bbox_inches="tight")
