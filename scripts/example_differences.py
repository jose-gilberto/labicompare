import numpy as np
import pandas as pd

from labicompare.core.data import EvaluationData
from labicompare.plots.differences import plot_difference_distribution
from labicompare.stats.pairwise import paired_ttest

# Simulando dados de 30 datasets
np.random.seed(10)
df = pd.read_csv('./results.csv', index_col='dataset')

dados = EvaluationData(df, higher_is_better=True)

# 1. Calculamos o teste estatístico
res_t = paired_ttest(dados, "InceptionTime", "FCN")
print(f"P-Valor do T-Test: {res_t.p_value:.4f}")

# 2. Geramos o plot
figura = plot_difference_distribution(dados, "InceptionTime", "FCN", figsize=(9, 4))
figura.savefig("distribuicao_diferencas.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo como 'distribuicao_diferencas.png'!")