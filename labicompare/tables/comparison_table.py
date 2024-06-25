import pandas as pd


def generate_comparison_table(
    metrics: pd.DataFrame,
    caption: str,
    label: str,
    position_preference: str = 'ht',
    full_page_width: bool = False,
    bold_best_result: bool = True,
    allow_draw: bool = False,
    precision_case: int = 3
) -> str:
    columns_num = metrics.columns.shape[-1]
    columns_align = 'l' + ('c' * (columns_num))
    columns_name = '\\textbf{Dataset} & ' + ' & '.join(f'\\textbf{{{model}}}' for model in metrics.columns)

    header = f"""
    \\begin{{{'table' if not full_page_width else 'table*'}}}[{position_preference}]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{{columns_align}}}
            \\hline
            {columns_name} \\\\
    """
    
    # Detecting the max value for each row
    max_values = metrics.idxmax(axis=1)
    for index, value in zip(max_values.index, max_values.values):
        metrics.loc[index][value] = f'\\textbf{{{metrics.loc[index][value]}}}'
        print(metrics.loc[index][value])
        
    
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


if __name__ == '__main__':
    df = pd.DataFrame({
        'KNN': [0.1, 0.9, 0.2, 0.5],
        'IsolationForest': [0.6, 0.2, 0.4, 0.5]
    }, index=['FordA', 'Adiac', 'ECG200', 'BeetleFly'])
    
    print(generate_comparison_table(
        df,
        caption='Example of Caption',
        label='example_label',
        full_page_width=True
    ))