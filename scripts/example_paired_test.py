import pandas as pd

from labicompare.core.data import EvaluationData
from labicompare.stats.pairwise import paired_ttest, sign_test, wilcoxon_signed_rank

data = pd.read_csv('./results.csv', index_col='dataset')
data = EvaluationData(data)

res_t = paired_ttest(data, "InceptionTime", "FCN")
res_sign = sign_test(data, "InceptionTime", "ROCKET")
res_wilcox = wilcoxon_signed_rank(data, "FCN", "LITETime")

print(f"T-Test   : p-value={res_t.p_value:.4f} | Vencedor={res_t.winner}")
print(f"Sign Test: p-value={res_sign.p_value:.4f} | Vencedor={res_sign.winner}")
print(f"Wilcoxon : p-value={res_wilcox.p_value:.4f} | Vencedor={res_wilcox.winner}")