import pandas as pd

from labicompare.core.data import EvaluationData
from labicompare.plots.heatmap import plot_pvalue_matrix
from labicompare.stats.posthoc import wilcoxon_holm

df = pd.read_csv("./results.csv", index_col="dataset")

dados = EvaluationData(df, higher_is_better=True)

summary = wilcoxon_holm(dados, alpha=0.05)

fig = plot_pvalue_matrix(summary, figsize=(14, 10))
fig.savefig("example_heatmap.png", dpi=300, bbox_inches="tight")

print("\n--- Automated Ranking ---")
print(summary.get_leaderboard())