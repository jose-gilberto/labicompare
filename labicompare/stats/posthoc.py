
import numpy as np
import scipy.stats as st

from labicompare.core.data import EvaluationData
from labicompare.core.results import ComparisonSummary, PairwiseResult
from labicompare.stats.friedman import friedman_test


def wilcoxon_holm(
    data: EvaluationData,
    alpha: float = 0.05
) -> ComparisonSummary:
    """
    Returns the p_values for each pairwise comparison executed between the models 
    provided in the dataset for Wilcoxon pairwise test with Holm alpha correction.
    
    The method uses a Friedman test to reject the null hypothesis of the entire 
    models being statistically equivalent before proceeding.

    Args:
        data (EvaluationData): instance containing all results data.
        alpha (float): Significance level (default: 0.05).

    Returns:
        p_values: Tuple list (model_1, model_2, p_value, is_significant).
            - model_1 and model_2: Name of each compared model.
            - p_value: p-value for Wilcoxon test.
            - is_significant: Bool indicating if its significant relevant 
              after Holm correction.
        m: Number of compared models.
    """

    f_stat, f_p = friedman_test(data)
    is_global_sig = f_p <= alpha

    if not is_global_sig:
        raise ValueError(
            f"The null-hypothesis of Friedman test cannot be rejected "
            f"(p-value: {f_p:.4f} > {alpha})."
        )
    
    models = data.model_names
    model_means = data._df.mean().to_dict()
    pairwise_list = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            perf1, perf2 = data._df[m1].values, data._df[m2].values
            
            p_val = float(st.wilcoxon(perf1, perf2, zero_method='pratt').pvalue)
            mean_diff = float(np.mean(perf1 - perf2))

            winner = None
            if mean_diff != 0:
                if data.higher_is_better:
                    winner = m1 if mean_diff > 0 else m2
                else:
                    winner = m1 if mean_diff < 0 else m2

            pairwise_list.append(PairwiseResult(
                model_a=m1, model_b=m2, p_value=p_val,
                is_significant=False, winner=winner, mean_diff=mean_diff
            ))

    pairwise_list.sort(key=lambda x: x.p_value)
    k = len(pairwise_list)
    for i in range(k):
        if pairwise_list[i].p_value <= (alpha / (k - i)):
            pairwise_list[i].is_significant = True
        else:
            break
        
    return ComparisonSummary(
        friedman_stat=f_stat,
        friedman_p_value=f_p,
        is_global_sig=is_global_sig,
        pairwise_results=pairwise_list,
        model_means=model_means,
        alpha=alpha,
        higher_is_better=data.higher_is_better,
        n_samples=len(data._df)
    )
