import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


from typing import Tuple
import matplotlib.pyplot as plt

def one_vs_one_plot(
    metrics_model_1: np.ndarray,
    metrics_model_2: np.ndarray,
    model_1_name: str = 'Model 1',
    model_2_name: str = 'Model 2',
    figsize: Tuple[float, float] = (6.4, 4.8),
    metric_name: str = 'performance',
    show_region_names: bool = True,
    legend_title: str = None,
    results_colors = ['blue', 'green', 'red']
):
    """
    Returns the One-versus-One plot between the two provided classifiers
    using the specific metric provided in the arrays.

    Args:
        metrics_model_1 (np.ndarray): metrics of model 1 in each dataset.
            The shape of this array must be (1, N).
        metrics_model_2 (np.ndarray): metrics of model 2 in each dataset.
            The shape of this array must be (1, N).
        model_1_name (str): name that will be used for model 1 when
            generating the plot.
        model_2_name (str): name that will be used for model 2 when
            generating the plot.
        figsize (tuple[float, float]): matplotlib figsize parameter.
        metric_name (str): name of the metric used to compare the two models.
        show_region_names (bool): show the name of the models in each 'winning'
            region of the plot.
        legend_title (str): matplotlib legend box title parameter.
        results_colors (tuple[str, str, str]): the colors that will represent
            the wins, tie, and losses. Each color must be compatible with
            the matplotlib colors.

    Returns:
        fig (matplotlib.figure.Figure): the object of the figure containing all
            components created to generate the one-versus-one plot.
        ax (matplotlib.pyplot.axes): last axes used to create the one-versus-one
            plot.

    Refs:
        [1] Ismail-Fawaz, Ali, et al. "Lite: Light inception with boosting techniques for time series classification." 2023 IEEE 10th International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2023.
    """
    if not isinstance(metrics_model_1, np.ndarray):
        metrics_model_1 = np.array(metrics_model_1)
    if not isinstance(metrics_model_2, np.ndarray):
        metrics_model_2 = np.array(metrics_model_2)

    assert metrics_model_1.shape != ()
    assert metrics_model_2.shape != ()

    assert metrics_model_1.shape == metrics_model_2.shape

    assert isinstance(figsize, tuple)
    assert isinstance(model_1_name, str) and isinstance(model_2_name, str)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(f'One vs One Comparison: {model_1_name} vs {model_2_name}')

    # Draw line
    ax.plot([0, 1], [0, 1], '--', color='gray')

    # Plot lost cases
    loss_counts = np.array([metrics_model_1[metrics_model_1 < metrics_model_2]]).shape[-1]

    ax.scatter(
        metrics_model_1[metrics_model_1 < metrics_model_2],
        metrics_model_2[metrics_model_1 < metrics_model_2],
        c='red', alpha=0.5,
        label=f'{loss_counts} losses'
    )

    # Plot draw cases
    draw_counts = np.array([metrics_model_1[metrics_model_1 == metrics_model_2]]).shape[-1]

    ax.scatter(
        metrics_model_1[metrics_model_1 == metrics_model_2],
        metrics_model_2[metrics_model_1 == metrics_model_2],
        c='green', alpha=0.5,
        label=f'{draw_counts} draws'
    )

    # Plot win cases
    win_counts = np.array([metrics_model_1[metrics_model_1 > metrics_model_2]]).shape[-1]

    ax.scatter(
        metrics_model_1[metrics_model_1 > metrics_model_2],
        metrics_model_2[metrics_model_1 > metrics_model_2],
        c='blue', alpha=0.5,
        label=f'{win_counts} wins'
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title=None, loc='lower right')

    ax.text(
        0.55, 0.3,
        f'{model_1_name} is\nbetter here', alpha=0.3,
        fontsize=18
    )

    ax.text(
        0.1, 0.6,
        f'{model_2_name} is\nbetter here', alpha=0.3,
        fontsize=18
    )

    ax.set_xlim([0., 1.])
    ax.set_xlabel(f'{model_1_name} {metric_name}')
    ax.set_ylim([0., 1.])
    ax.set_ylabel(f'{model_2_name} {metric_name}')

    return fig, ax