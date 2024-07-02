import pandas as pd
import numpy as np


def generate_one_vs_one_table(
    metrics: pd.DataFrame,
    caption: str,
    label: str,
    model_1: str = 'Model 1',
    model_2: str = 'Model 2',
    dataset_repository: str = 'All Datasets',
    position_preference: str = 'ht',
    mean_difference_precision: int = 4,
) -> str:
    # Filter only the two models that will be compared
    metrics = metrics[[model_1, model_2]]

    header = f"""
    \\begin{{table}}[{position_preference}]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{lc}}
            \\hline
             & \\textbf{{{dataset_repository}}} \\\\
            \\hline\\hline
    """
    
    content = ""
    
    # Paired t-test p-value

    # Mean Difference
    model_1_mean = metrics[model_1].mean()
    model_2_mean = metrics[model_2].mean()
    mean_difference = model_1_mean - model_2_mean
    
    content += f'\\textbf{{Mean Difference}} & {round(mean_difference, mean_difference_precision)} \\\\'
    
    # Wins_Draws_Looses
    
    footer = f"""
        \\hline
        \\end{{tabular}}
    \\end{{table}}
    """

    table = header + content + footer

    return table