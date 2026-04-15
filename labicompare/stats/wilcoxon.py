import scipy.stats as st

from labicompare.core.data import EvaluationData


def wilcoxon_test(
    data: EvaluationData, model_1: str | None = None, model_2: str | None = None
) -> tuple[float, float]:
    """
    Runs posthoc Wilcoxon test for two models.

    If EvaluationData contains only two models, model_1 and model_2 are
    optional parameters. Otherwise, the name for the models must be informed.

    Args:
        data: EvaluationData instance containing all results.
        model_1: name of the first model.
        model_2: name of the second model.

    Returns:
        a tuple containing (stat_w, p_value)
    """
    n_models = len(data.model_names)

    if model_1 is None or model_2 is None:
        if n_models == 2:
            model_1, model_2 = data.model_names
        else:
            raise ValueError(
                "EvaluationData contains more than 2 models. You must specify "
                "model_1 and model_2."
            )
        
    if model_1 not in data.model_names or model_2 not in data.model_names:
        raise ValueError(
            f"Models not found! Available models {data.model_names}."
        )
    
    if model_1 == model_2:
        raise ValueError("Models for comparison must be different!")
    
    scores_1 = data._df[model_1].values
    scores_2 = data._df[model_2].values

    differences = scores_1 - scores_2

    if all(d == 0 for d in differences):
        return 0.0, 1.0
    
    # Perform the test. Use an alternative "two-sided" to check each difference
    # zero_method="wilcox" discard differences equals to 0
    w_stat, p_value = st.wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided"
    )
    return float(w_stat), float(p_value)
