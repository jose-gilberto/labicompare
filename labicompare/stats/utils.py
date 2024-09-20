import pandas as pd


def calculate_average_ranks(
    metrics: pd.DataFrame,
    tie_method: str = 'average',
) -> pd.Series:
    """
    Calculate the mean average ranks for each model over multiple
    datasets.

    Args:
        metrics (pd.DataFrame): dataframe where columns represent each model
            and lines contains the metrics for each dataset.
    
    Returns:
        pd.Series: mean ranking for each model.
    """
    ranks = metrics.rank(axis=1, method=tie_method, ascending=False)
    average_ranks = ranks.mean(axis=0)

    return average_ranks
