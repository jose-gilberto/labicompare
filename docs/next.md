# Roadmap & Next Steps

**LabiCompare** is continuously evolving. While our current focus is on providing a rock-solid implementation of the Friedman + Wilcoxon-Holm pipeline, we have several exciting features planned for future releases. 

If you are a developer looking to contribute, these areas are great places to start.

## 1. Expanded Statistical Methods

While the Wilcoxon-Holm procedure is highly recommended for ML benchmarks, different research domains sometimes expect alternative tests. We plan to expand our statistical engine to include:

* **Iman-Davenport Extension:** A less conservative alternative to the standard Friedman test, often preferred when the number of datasets is small.
* **Nemenyi Post-Hoc Test:** A widely known alternative for pairwise comparisons. Although Wilcoxon-Holm is generally more powerful, supporting Nemenyi allows users to replicate older benchmark papers.
* **Bayesian Correlated t-Tests:** Moving beyond frequentist p-values to provide probabilities of one model being practically better than another (e.g., using the Region of Practical Equivalence - ROPE).

## 2. Advanced Visualizations

The Critical Difference (CD) diagram is our flagship plot, but sometimes different perspectives are needed to fully understand model behavior.

* **P-Value Heatmaps:** A lower-triangular heatmap showing the adjusted p-values between all pairs of models, making it easy to spot clusters of similarly performing algorithms.
* **Win/Tie/Loss Matrices:** A graphical representation of how many datasets Model A won, tied, or lost against Model B, providing an intuitive, non-statistical summary of relative performance.
* **Rank Boxplots:** Visualizing the distribution of ranks for each model across all datasets to highlight variance and outliers.

and **Multi-Comparison Matrix** as well.

## 3. Developer Experience & Integrations

We want to make `labicompare` as frictionless as possible within existing ML workflows.

* **Command-Line Interface (CLI):** Allow users to run statistical tests and generate plots directly from the terminal without writing Python scripts.
    ```bash
    # Planned feature
    labicompare run results.csv --higher-better --plot-cd
    ```
* **Scikit-Learn Integration:** Native support for parsing the output of `sklearn.model_selection.cross_validate` directly into our `EvaluationData` object.
* **Automated Report Generation:** Export a complete HTML or PDF report containing the Friedman global results, the p-value tables, and the CD diagram in one step.

and also an **INCREDIBLE SUPPORT TO LATEX TABLES AND TEMPLATES :)**

---

## Want to help?

If you are interested in tackling any of these features, please check our [Contributing Guidelines](index.md#contributing) and open an issue or pull request on our GitHub repository. Help me to finish this idea, please!
