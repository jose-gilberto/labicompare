import matplotlib.pyplot as plt
import pandas as pd

from labicompare.stats import calculate_average_ranks, nemenyi, bonferroni_dunn, wilcoxon_holm
from .utils import _graph_ranks, _wilcoxon_graph_ranks


def diagram_nemenyi(
    avg_ranks: pd.Series,
    n_experiments: int,
    alpha: float = 0.05
):
    cd = nemenyi(avg_ranks, n_experiments, alpha=alpha)
    fig = _graph_ranks(
        avranks=avg_ranks.values,
        names=avg_ranks.index, cd=cd
    )
    return fig


def diagram_bonferroni(
    avg_ranks: pd.Series,
    n_experiments: int,
    alpha: float = 0.05
):
    cd = bonferroni_dunn(avg_ranks, n_experiments, alpha=alpha)
    fig = _graph_ranks(
        avranks=avg_ranks.values,
        names=avg_ranks.index, cd=cd, cdmethod=0
    )
    return fig

def diagram_wilcoxon(
    metrics: pd.DataFrame,
    alpha: float = 0.05,
):
    p_values, _ = wilcoxon_holm(metrics, alpha=alpha)
    avg_ranks = calculate_average_ranks(metrics).sort_values(ascending=False)

    # TODO: refactor the wilcoxon graph ranks to use the same graph ranks
    # function as nemenyi and bonferroni
    fig = _wilcoxon_graph_ranks(
        avg_ranks.values,
        avg_ranks.keys(),
        p_values,
        cd=None,
        reverse=True,
        width=9,
        textspace=1.5,
        labels=False
    )

    return fig


def critical_difference_diagram(
    metrics: pd.DataFrame,
    test: str,
    alpha: float = 0.05
):
    num_experiments = metrics.shape[0]
    avg_ranks = calculate_average_ranks(metrics)

    if test == 'nemenyi':
        return diagram_nemenyi(
            avg_ranks=avg_ranks, n_experiments=num_experiments, alpha=alpha
        )
    if test == 'bonferroni':
        return diagram_bonferroni(
            avg_ranks=avg_ranks, n_experiments=num_experiments, alpha=alpha
        )
    if test == 'wilcoxon':
        return diagram_wilcoxon(metrics, alpha=alpha)
    else:
        raise NotImplemented()