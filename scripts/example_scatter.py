import pandas as pd

from labicompare.core.data import EvaluationData
from labicompare.plots.scatter import plot_one_vs_one

df = pd.read_csv("./results.csv", index_col='dataset')

data = EvaluationData(df, higher_is_better=True)

fig_scatter = plot_one_vs_one(
  data,
  model_x="InceptionTime",
  model_y="FCN",
  figsize=(12, 8)
)
fig_scatter.savefig("1v1_example.png", dpi=300, bbox_inches="tight")
