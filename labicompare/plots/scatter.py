import matplotlib.pyplot as plt
import numpy as np

from labicompare.core.data import EvaluationData


def plot_one_vs_one(
  data: EvaluationData,
  model_x: str,
  model_y: str,
  figsize: tuple[float, float] = (6, 6),
  point_size: int = 50,
  alpha_points: float = 0.7
) -> plt.Figure:
  """
  Generates a scatter plot comparing the results of two specific models
  dataset per dataset. Diagonal line represents draws (y = x).
  
  Args:
    data: EvaluationData containing all results.
    model_x: Name of the model that will be represented in the X-axis.
    model_y: Name fo the model that will be represented in the Y-axis.
    figsize: Figure size.
    point_size: Point size used in scatter plot.
    alpha_points: Point transparency (useful for overlays).
      
  Returns:
    Matplotlib figure.
  """
  if model_x not in data.model_names or model_y not in data.model_names:
    raise ValueError(f"Models '{model_x}' or '{model_y}' not found in data.")

  perf_x = data._df[model_x].values
  perf_y = data._df[model_y].values

  fig, ax = plt.subplots(figsize=figsize)

  min_val = min(perf_x.min(), perf_y.min())
  max_val = max(perf_x.max(), perf_y.max())

  padding = (max_val - min_val) * 0.05
  if padding == 0:
    padding = 0.1
      
  lim_min: float = min_val - padding
  lim_max: float = max_val + padding

  ax.plot(
    [lim_min, lim_max], [lim_min, lim_max], 
    color="gray", linestyle="--", zorder=1, label="Draw-line (y = x)"
  )

  diff = perf_x - perf_y
  if data.higher_is_better:
    x_wins = diff > 0
    y_wins = diff < 0
  else:
    x_wins = diff < 0
    y_wins = diff > 0
  ties = diff == 0

  if np.any(x_wins):
    ax.scatter(
      perf_x[x_wins], perf_y[x_wins], 
      color="#2ca02c", s=point_size, alpha=alpha_points, 
      zorder=2, label=f"{model_x} wins ({len(perf_x[x_wins])} datasets)"
    )
      
  if np.any(y_wins):
      ax.scatter(
          perf_x[y_wins], perf_y[y_wins], 
          color="#d62728", s=point_size, alpha=alpha_points, 
          zorder=2, label=f"{model_y} wins ({len(perf_x[y_wins])} datasets)"
      )
  
  if np.any(ties):
      ax.scatter(
          perf_x[ties], perf_y[ties], 
          color="black", s=point_size, marker="x",
          zorder=3, label=f"Exact draws ({len(perf_x[ties])} datasets)"
      )

  ax.set_xlim((lim_min, lim_max))
  ax.set_ylim((lim_min, lim_max))
  ax.set_aspect("equal", adjustable="box")

  ax.set_xlabel(f"Performance: {model_x}", fontweight="bold", labelpad=10)
  ax.set_ylabel(f"Performance: {model_y}", fontweight="bold", labelpad=10)
  ax.set_title(f"One versus One Plot between \n{model_x} vs {model_y}", pad=15)
  
  ax.grid(True, linestyle=":", alpha=0.6)
  ax.legend(loc="best", framealpha=0.9)
  
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  fig.tight_layout()
  
  return fig