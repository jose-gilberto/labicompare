import matplotlib.pyplot as plt
import numpy as np

from labicompare.core.data import EvaluationData


def plot_difference_distribution(
  data: EvaluationData,
  model_a: str,
  model_b: str,
  figsize: tuple[float, float] = (10, 4)
) -> plt.Figure:
  """
  Generates a horizontal boxplot combined with a scatter plot (jitter)
  focused on difference between two models. The central line (zero) divides
  visually who wins in the majority of the samples.
  """
  if model_a not in data.model_names or model_b not in data.model_names:
    raise ValueError(f"The Models '{model_a}' or '{model_b}' not found in results.")

  diffs = data._df[model_a] - data._df[model_b]
  mean_diff = diffs.mean()
  median_diff = diffs.median()
  
  fig, ax = plt.subplots(figsize=figsize)
  
  max_abs_diff = max(abs(diffs.min()), abs(diffs.max()))
  limit = max_abs_diff * 1.15 
  
  color_win_a = "#10b981" # Emerald Green
  color_win_b = "#ef4444" # Coral Red
  color_tie = "#9ca3af"   # Cool Gray

  if data.higher_is_better:
    point_colors = np.where(diffs > 0,
                            color_win_a,
                            np.where(diffs < 0, color_win_b, color_tie))
  else:
    point_colors = np.where(diffs < 0,
                            color_win_a,
                            np.where(diffs > 0,color_win_b, color_tie))

  ax.axvline(0, color='#6b7280', linestyle='-', linewidth=1.5, zorder=1)
  
  boxprops = dict(facecolor="#f3f4f6", color="#9ca3af", linewidth=1.5, alpha=0.5)
  medianprops = dict(color="#374151", linewidth=2.5)
  whiskerprops = dict(color="#9ca3af", linewidth=1.5, linestyle="--")
  capprops = dict(color="#9ca3af", linewidth=1.5)

  _ = ax.boxplot(
    diffs, 
    vert=False, 
    patch_artist=True, 
    widths=0.3,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    showfliers=False,
    zorder=2
  )
  
  np.random.seed(42)
  y_jitter = np.random.normal(1, 0.06, size=len(diffs))
  
  ax.scatter(
    diffs, 
    y_jitter, 
    c=point_colors, 
    linewidth=0.8,
    s=30,
    alpha=0.4,
    zorder=3
  )

  ax.plot(mean_diff, 1, marker='D', color='black', markersize=4, zorder=4, label='Mean')

  label_right = (f"Favors {model_a} →" if data.higher_is_better
                 else f"Favors {model_b} →")

  ax.text(limit*0.05, 1.4, label_right,
          color=color_win_a if data.higher_is_better else color_win_b, 
          fontsize=10, fontweight='bold', va='center', ha='left')
  
  label_left = f"← Favors {model_b}" if data.higher_is_better else f"← Favors {model_a}"
  ax.text(-limit*0.05, 1.4, label_left,
          color=color_win_b if data.higher_is_better else color_win_a, 
          fontsize=10, fontweight='bold', va='center', ha='right')

  stats_text = f"Mean: {mean_diff:+.4f}\nMedian: {median_diff:+.4f}"
  ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
          ha='right', va='bottom',
          bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='white', alpha=0.9,
                    edgecolor='#d1d5db'))

  ax.set_xlim(-limit, limit)
  ax.set_ylim(0.5, 1.5)
  ax.set_yticks([]) 
  
  ax.set_xlabel(f"Absolute Difference ({model_a} - {model_b})",
                fontweight='bold', labelpad=10, color="#374151")
  ax.set_title("Distribution of Differences", pad=20,
               fontsize=14, fontweight='bold', color="#111827")

  ax.grid(axis='x', color="#e5e7eb", linestyle='--', linewidth=1, zorder=0)

  for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
  ax.spines['bottom'].set_color('#9ca3af')

  fig.tight_layout()
  
  return fig