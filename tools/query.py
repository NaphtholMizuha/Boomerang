import polars as pl
from tabulate import tabulate
import json

def gen_query(groups, condition):
    select_cols = ['params']
    for key, alias in groups.items():
        select_cols.append(f"params#>>'{{{key}}}' AS {alias}")
    select_clause = ','.join(select_cols)
    
    return f"""
    SELECT r.*, {select_clause}
    FROM results r
    JOIN params p ON r.id = p.id
    WHERE p.params @> '{json.dumps(condition)}'::jsonb
    """

# Set the maximum number of rows to display in Polars tables
pl.Config.set_tbl_rows(1000)

condition = {
    'split': {
        'method': 'iid'
    },
    'local': {
        'model': 'resnet'
    },
    'learner': {
        'm': 2,
        'defense': 'score'
    },
    'aggregator': {
        'm': 1,
    }
    
}

# Define the URI for the SQLite database
uri = "postgresql://wuzihou:wuzihou@localhost:5432/feddpr"

groups = {
    # 'aggregator,attack': 'atk_agg',
    # 'aggregator,defense': 'def_agg',
    # 'learner,defense': 'def_lrn',
    # 'local,model': 'model'
    'penalty': 'penalty'
}
    
query = gen_query(groups, condition)
# query = "SELECT * FROM results ORDER BY (id, turn, rnd)"

# Read the data from the database into a Polars DataFrame
df = pl.read_database_uri(query=query, uri=uri)

# Print the shape of the DataFrame to verify the data was loaded correctly
print(df)

# # Add a new column 'm_rate' which is the ratio of 'm_lrn' to 'n_lrn', then drop the original columns
# df = df.with_columns(
#         (pl.col("m_lrn") / pl.col("n_lrn")).alias("mal_rate")
#     ).drop(["m_lrn", "n_lrn"])
aliases = [value for value in groups.values()]
# Group the DataFrame by 'm_rate' and 'turn', then calculate the minimum loss and maximum accuracy for each group
df = df.group_by(["id"] + aliases + ["turn"]).agg([
    pl.col("loss").min().alias("min_loss"),
    pl.col("acc").max().alias("max_acc")
# Group the resulting DataFrame by 'm_rate' and calculate the mean of the minimum loss and maximum accuracy, rounding to 2 decimal places
]).group_by(["id"] + aliases).agg([
    pl.col("min_loss").mean().round(2).alias("mean_min_loss"), # round to 2 decimal places
    pl.col("max_acc").mean().mul(100).round(2).alias("mean_max_acc")
# Sort the DataFrame by 'm_rate'
]).sort('id')

# Print the final DataFrame
print(f'Query with:\n{condition}\n{df}')

# # Transpose the DataFrame to prepare it for display as a markdown table
# df = df.transpose(include_header=True, header_name="m_rate")

# # Print the transposed DataFrame as a markdown table using the 'tabulate' library
# print(tabulate(df, headers="firstrow", tablefmt="github"))

# def df_to_latex(df: pl.DataFrame, caption: str = "", label: str = "") -> str:
#     """
#     Convert a Polars DataFrame to a LaTeX table
    
#     Args:
#         df: Polars DataFrame to convert
#         caption: Table caption
#         label: Table label for referencing
        
#     Returns:
#         str: LaTeX table code
#     """
#     # Convert DataFrame to list of lists for tabulate
#     data = df.transpose(include_header=True, header_name="").rows()
    
#     # Generate LaTeX table using tabulate
#     latex_table = tabulate(data, headers="firstrow", tablefmt="latex")
    
#     # Add caption and label if provided
#     if caption:
#         latex_table = latex_table.replace("\\begin{tabular}", 
#             f"\\begin{{table}}[htbp]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
#         latex_table += "\n\\end{table}"
    
#     return latex_table

# # Example usage:
# latex_table = df_to_latex(df, caption="Model Performance", label="tab:model_perf")
# print(latex_table)
