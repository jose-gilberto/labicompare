import pandas as pd

from labicompare.core.data import EvaluationData
from labicompare.plots.ranking import plot_cd_diagram
from labicompare.stats.posthoc import wilcoxon_holm

df = pd.read_csv("./results.csv", index_col="dataset")

data = EvaluationData(df, higher_is_better=True)

summary = wilcoxon_holm(data, alpha=0.05)
fig = plot_cd_diagram(data, summary, highlight_models=['InceptionTime'])

fig.savefig("cd_diagram.png", dpi=300)