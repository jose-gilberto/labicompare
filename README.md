# LabiCompare

LabiCompare is a Python library focused on the evaluation and statistical comparison of Machine Learning models. It simplifies the process of running hypothesis tests (such as Friedman and Wilcoxon-Holm) and generates visualizations, including Critical Difference (CD) Diagrams.

## Instalation

Via PyPI (future - WIP):

```bash
pip install labicompare
```

Or install it locally from the source code:

```bash
git clone https://github.com/jose-gilberto/labicompare.git
cd labicompare
pip install -e .
```

## Quick Start

Here is how easy it is to compare the performance of multiple models across different datasets or folds:
```python
import pandas as pd
from labicompare.core.data import EvaluationData

from labicompare.stats import evaluate_models 
from labicompare.plots.ranking import plot_cd_diagram

# 1. Prepare your data (Rows = Datasets/Folds, Columns = Models)
data_dict = {
    'Model_A': [0.85, 0.88, 0.82, 0.89],
    'Model_B': [0.86, 0.89, 0.84, 0.90],
    'Model_C': [0.70, 0.75, 0.72, 0.71],
    'Proposed_Model': [0.91, 0.93, 0.89, 0.95]
}
df = pd.DataFrame(data_dict)

# 2. Wrap the data (Accuracy: higher_is_better=True)
eval_data = EvaluationData(df, higher_is_better=True)

# 3. Run the statistical tests (e.g., Friedman + Wilcoxon-Holm)
summary = wilcoxon_holm(eval_data, alpha=0.05)

print(summary)
# Output: ComparisonSummary(Friedman P-Value=0.0012, H0=REJECTED, Models=4)

# 4. Generate the Critical Difference Diagram (CD Diagram)
fig = plot_cd_diagram(
    data=eval_data,
    summary=summary,
    title="Model Comparison (Accuracy)"
)
fig.savefig("cd_diagram.png", dpi=300, bbox_inches='tight')
```

## Core Components

### 1. `EvaluationData`

The base class that ingests your `pandas.DataFrame`.

Key Parameter: `higher_is_better` (Boolean). Use `True` for metrics like Accuracy and F1-Score, and `False` for error metrics like RMSE or MAE. The library automatically handles ranking inversions under the hood.

### 2. `ComparisonSummary`

The object returned by the statistical testing functions. It stores:
- Global results (Friedman's P-value).
- Pairwise results (`pairwise_results`), including which model won and whether the difference is statistically significant.
- Built-in Export: You can use `summary.to_dataframe()` to export the results into a tabular format, making it easy to convert to LaTeX or Markdown for your papers.

### 3. Critical Difference Diagrams (`plot_cd_diagram`)

The visual tool for comparing models. Our implementation features an enhanced UX tailored for academic publishing:

- Bilateral Layout: Models are split evenly on both sides to prevent text overlap.
- Maximal Cliques: Thick bars group models that have no statistically significant difference (ties), automatically preventing redundant sub-lines.
- Inline Rankings: The exact average rank is displayed cleanly beneath each model's name.

#### Highlighting Your Model

If you are proposing a new model and want it to stand out in the diagram, use the highlight parameters:

```python
fig = plot_cd_diagram(
    data=eval_data,
    summary=summary,
    highlight_models=['Proposed_Model'],
    highlight_color='#d97706' # Optional custom color (Default: Amber/Orange)
)
```

## Contributing

We welcome contributions from the community! Whether you want to fix a bug, add a new statistical test, or improve the documentation, your help is highly appreciated.

### Development Setup

1. Fork the repository and clone it locally:
    ```bash
    git clone [https://github.com/your-username/labicompare.git](https://github.com/your-username/labicompare.git)
    cd labicompare
    ```

2. Create a virtual environment and install the development dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -e ".[dev]"   # Ensure you have a [dev] extra in your pyproject.toml or setup.py
    ```

#### Testing

All new features and bug fixes should be accompanied by unit tests. We use pytest as our testing framework.

To run the test suite:
```bash
pytest tests/
```

#### Pull Request Process

Create a new branch for your feature or bugfix (`git checkout -b feature/my-awesome-feature`).

Make your changes and commit them with descriptive messages.

Ensure all tests pass and the code is properly formatted.

Push your branch to your fork (`git push origin feature/my-awesome-feature`).

Open a Pull Request against the main branch of this repository. Include a clear description of the changes and any related issue numbers.

#### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Provide as much detail as possible, including steps to reproduce bugs or a clear rationale for new features.

## 🎓 Citation

If you use **labicompare** in your research or project, please consider citing it.

```bibtex
@misc{labicompare2026,
  author       = {José Gilberto Barbosa de Medeiros Júnior},
  title        = {labicompare: statistical comparison and visualization for Machine Learning models},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/jose-gilberto/labicompare](https://github.com/jose-gilberto/labicompare)}},
}
```
