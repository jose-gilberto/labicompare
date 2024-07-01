import pandas as pd


def generate_comparison_table(
    metrics: pd.DataFrame,
    caption: str,
    label: str,
    position_preference: str = 'ht',
    full_page_width: bool = False,
    bold_best_result: bool = True,
    allow_draw: bool = True,
    precision_case: int = 3
) -> str:
    """
    """
    columns_num = metrics.columns.shape[-1]
    columns_align = 'l|' + ('c' * (columns_num))
    columns_name = '\\textbf{Dataset} & ' + ' & '.join(f'\\textbf{{{model}}}' for model in metrics.columns)

    header = f"""
    \\begin{{{'table' if not full_page_width else 'table*'}}}[{position_preference}]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{{columns_align}}}
            \\hline
            {columns_name} \\\\
            \\hline\\hline
    """

    if bold_best_result:
        # Detecting the max value for each row
        maximum_values = metrics.max(axis=1)
        for i, row in metrics.iterrows():
            for column_name, column_value in row.items():
                count = -2 if allow_draw else -1
                if maximum_values[i] <= column_value and count < 0:
                    metrics.at[i, column_name] = f'\\textbf{{{column_value}}}'
                    count += 1


    nl = '\n            '
    endline = ' \\\\'
    content = f"""        {nl.join((metrics.index[i] + ' & ' + ' & '.join(str(metric) for metric in line) + endline) for i, line in enumerate(metrics.values))}"""
    
    footer = f"""
        \\hline
        \\end{{tabular}}
    \\end{{{'table' if not full_page_width else 'table*'}}}
    """
    
    table = header + content + footer

    return table
