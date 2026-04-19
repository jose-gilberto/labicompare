from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from labicompare.core.results import ComparisonSummary


def plot_pvalue_matrix(
    summary: ComparisonSummary,
    figsize: tuple[float, float] = (8, 6),
    fontsize: int = 12,
    grid_linewidth: float = 2.0,
    aspect: Literal["equal", "auto"] | float | None = "equal"
) -> plt.Figure:
    """
    Generate a styled heatmap for p-values matrix from a post-hoc test.

    Args:
        pvalue_matrix: DataFrame with pair-wise p-values.
        alpha: Significance level (default: 0.05). Values <= alpha will be in bold.
        min_alpha: The exact value where the color changes to green (default: 0.10).
        figsize: Size of the figure (width, height).
        fontsize: Size of the font used in this plot.
        grid_linewidth: Width for the grid line used between cells.

    Returns:
        Figure instance from matplotlib, can be saved or showed.
    """
    fig, ax = plt.subplots(figsize=figsize)


    model_means = summary.model_means
    higher_is_better = summary.higher_is_better
    base_alpha = summary.alpha
    
    models = sorted(
        list(model_means.keys()), 
        key=lambda m: model_means[m], 
        reverse=higher_is_better
    )
    n_models = len(models)

    data = np.full((n_models, n_models), np.nan)
    sig_matrix = np.full((n_models, n_models), False)
    winner_matrix = np.full((n_models, n_models), None, dtype=object)

    for res in summary.pairwise_results:
        i = models.index(res.model_a)
        j = models.index(res.model_b)
        
        data[i, j] = data[j, i] = res.p_value
        sig_matrix[i, j] = sig_matrix[j, i] = res.is_significant
        winner_matrix[i, j] = winner_matrix[j, i] = res.winner
        
        
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#f0f0f0") 
    
    norm = TwoSlopeNorm(vmin=0.0, vcenter=base_alpha, vmax=1.0)
    cax = ax.imshow(data, cmap=cmap, norm=norm, aspect=aspect)
    
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.set_ylabel("Adjusted P-value", rotation=-90, va="bottom", labelpad=15)

    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(models)
    ax.set_yticklabels(models)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                ax.text(j, i, "-",
                        ha="center", va="center",
                        color="gray", fontsize=fontsize)
            else:
                val = data[i, j]
                is_sig = sig_matrix[i, j]
                winner = winner_matrix[i, j]
                row_model = models[i]
                
                if is_sig:
                    indicator = " (↑)" if winner == row_model else " (↓)"
                    weight = "bold"
                    text_str = f"{val:.3f}{indicator}"
                else:
                    weight = "normal"
                    text_str = f"{val:.3f}"
                
                text_color = "white" if (val < (base_alpha/2) or val > 0.7) else "black"
                ax.text(j, i, text_str,
                        ha="center", va="center", color=text_color,
                        fontweight=weight, fontsize=fontsize)

    if grid_linewidth > 0:
        ax.set_xticks(np.arange(n_models + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_models + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=grid_linewidth)
        ax.tick_params(which="minor", bottom=False, left=False)

    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_title("Pairwise Comparison Matrix\n(Post-Hoc P-values)")

    fig.tight_layout()

    return fig