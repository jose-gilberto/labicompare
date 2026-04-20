import matplotlib.pyplot as plt
import numpy as np

from labicompare.core.data import EvaluationData
from labicompare.core.results import ComparisonSummary


def plot_cd_diagram(
  data: EvaluationData,
  summary: ComparisonSummary,
  title: str = "Critical Difference Diagram (Wilcoxon-Holm)",
  figsize: tuple[float, float] = (12, 5),
  highlight_models: list[str] | None = None,
  highlight_color: str = "#d97706"
) -> plt.Figure:

  ranks = data._df.rank(axis=1, ascending=not data.higher_is_better).mean()
  ranks = ranks.sort_values()
  model_names = ranks.index.tolist()
  avg_ranks = ranks.values
  n_models = len(model_names)

  fig, ax = plt.subplots(figsize=figsize)

  ax.set_xlim(0, n_models + 1)
  ax.set_ylim(-0.2, 1.2) 
  ax.set_yticks([])
  ax.set_xticks([])
  
  for spine in ['top', 'right', 'left', 'bottom']:
      ax.spines[spine].set_visible(False)
  
  # Main axis
  y_axis = 1.0
  ax.plot([1, n_models], [y_axis, y_axis], color='#000000', linewidth=2.5, zorder=1)
  
  for i in range(1, n_models + 1):
    ax.plot([i, i], [y_axis, y_axis + 0.03], color='#000000', linewidth=2.6)
    ax.text(i, y_axis + 0.05, str(i), ha='center', va='bottom', 
            fontsize=16, fontweight='bold', color='#1f2937')

  is_sig_matrix = np.zeros((n_models, n_models), dtype=bool)
  for res in summary.pairwise_results:
    if res.model_a in model_names and res.model_b in model_names:
      idx1 = model_names.index(res.model_a)
      idx2 = model_names.index(res.model_b)
      is_sig_matrix[idx1, idx2] = res.is_significant
      is_sig_matrix[idx2, idx1] = res.is_significant

  cliques = []
  for i in range(n_models):
    for j in range(i + 1, n_models):
      is_clique = True
      for a in range(i, j + 1):
        for b in range(a + 1, j + 1):
          if is_sig_matrix[a, b]:
            is_clique = False
            break
        if not is_clique:
          break
      if is_clique:
        cliques.append((i, j))

  maximal_cliques = []
  for c1 in cliques:
    is_maximal = True
    for c2 in cliques:
      if c1 != c2 and c1[0] >= c2[0] and c1[1] <= c2[1]:
        is_maximal = False
        break
    if is_maximal and c1 not in maximal_cliques:
      maximal_cliques.append(c1)

  y_ns_base = 0.92 
  ns_line_step = 0.06

  for idx, clique in enumerate(maximal_cliques):
      start_rank = avg_ranks[clique[0]]
      end_rank = avg_ranks[clique[1]]
      y_ns = y_ns_base - (idx * ns_line_step)
      
      ax.plot([start_rank, end_rank], [y_ns, y_ns], 
              color='#000000', linewidth=5, zorder=10)

  # 6. Calcular espaço para os nomes e desenhar conexões em "L"
  num_bars = len(maximal_cliques)
  lowest_ns_y = (
    y_ns_base - (max(0, num_bars - 1) * ns_line_step) if num_bars > 0 else y_axis
  )
  y_name_max = min(0.65, lowest_ns_y - 0.1) 
  
  split_idx = (n_models + 1) // 2
  y_space_left = np.linspace(y_name_max, 0.0, max(1, split_idx))
  y_space_right = np.linspace(y_name_max, 0.0, max(1, n_models - split_idx))

  color_line = '#000000'
  
  for i, (name, rank) in enumerate(zip(model_names, avg_ranks)):  
    is_highlighted = highlight_models is not None and name in highlight_models
    
    point_color = highlight_color if is_highlighted else '#1f77b4'
    text_color = highlight_color if is_highlighted else '#111827'

    ax.scatter(rank, y_axis, color=point_color,
               s=80, zorder=15, edgecolor='white', linewidth=1.5)

    if i < split_idx:
        x_name = 0.2
        ha = 'right'
        y_name = y_space_left[i]
    else:
        x_name = n_models + 0.8
        ha = 'left'
        y_name = y_space_right[i - split_idx]

    ax.plot([rank, rank], [y_axis, y_name], color=color_line, linewidth=1.5, zorder=2)
    ax.plot([rank, x_name], [y_name, y_name], color=color_line, linewidth=1.5, zorder=2)
    
    offset = 0.01
    ax.text(x_name, y_name + offset, name, ha=ha, va='bottom', 
            fontweight='bold', fontsize=11, color=text_color)
    ax.text(x_name, y_name - offset, f"({rank:.2f})", ha=ha, va='top', 
            fontsize=10, color='#000000')

  ax.set_title(title, pad=30, fontsize=14, fontweight='bold', color='#111827')
  plt.tight_layout()
  return fig