import polars as pl
from tabulate import tabulate
import json

def gen_query(group: list, condition: dict):
    select = ",".join(group)
    cond = ",".join([f"{key}={value}" for key, value in condition.items()])
    print(cond)
    return f"""
    SELECT r.*, {select}
    FROM result r
    JOIN config c on r.id = c.id
    WHERE {cond}
    """

def gen_json_query(groups, condition):
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
    'mal_rate_lrn': 0.25
}

# Define the URI for the SQLite database
uri = "postgresql://wuzihou:wuzihou@localhost:5432/feddpr"

groups = ['mal_rate_lrn', 'mal_rate_agg', 'attack_lrn', 'defense_lrn', 'attack_agg']
    
query = gen_query(groups, condition)
print(query)

# Read the data from the database into a Polars DataFrame
df = pl.read_database_uri(query=query, uri=uri)

# Print the shape of the DataFrame to verify the data was loaded correctly
# print(df)

# Group the DataFrame by 'm_rate' and 'turn', then calculate the minimum loss and maximum accuracy for each group
df = df.group_by(["id"] + groups + ["turn"]).agg([
    pl.col("loss").min().alias("min_loss"),
    pl.col("acc").max().alias("max_acc")
# Group the resulting DataFrame by 'm_rate' and calculate the mean of the minimum loss and maximum accuracy, rounding to 2 decimal places
]).group_by(["id"] + groups).agg([
    pl.col("min_loss").mean().round(2).alias("mean_min_loss"), # round to 2 decimal places
    pl.col("max_acc").mean().mul(100).round(2).alias("mean_max_acc")
# Sort the DataFrame by 'm_rate'
]).sort('id')

df = df.filter(~pl.col("id").is_in([33, 34, 30]))

df = df.with_columns(
    pl.when(pl.col("mal_rate_agg") == 0.0)
      .then(pl.lit("none"))
      .otherwise(pl.col("attack_agg"))
      .alias("attack_agg")
).drop(["mal_rate_agg", "mal_rate_lrn", "id"])

df = df.with_columns(
    pl.when((pl.col("defense_lrn") == pl.lit("none")) & (pl.col("attack_agg") == pl.lit("none")))
      .then(pl.lit("noattack"))
      .otherwise(pl.col("defense_lrn"))
      .alias("defense")
).drop(['defense_lrn', 'attack_agg', 'mean_min_loss'])


# Print the final DataFrame
print(f'Query with:\n{condition}\n{df}')

df_pivot = df.pivot(
    values="mean_max_acc",  # 需要填充的值
    index="attack_lrn",     # 作为行索引
    columns="defense",      # 作为列索引
    aggregate_function="first"  # 由于每个 (attack_lrn, defense) 组合只有一个值，用 first 即可
)

print(df_pivot)



def df_to_latex(df: pl.DataFrame, caption: str = "", label: str = "") -> str:
    """
    Convert a Polars DataFrame to a LaTeX table
    
    Args:
        df: Polars DataFrame to convert
        caption: Table caption
        label: Table label for referencing
        
    Returns:
        str: LaTeX table code
    """
    # Convert DataFrame to list of lists for tabulate
    data = df.transpose(include_header=True, header_name="").rows()
    
    # Generate LaTeX table using tabulate
    latex_table = tabulate(data, headers="firstrow", tablefmt="latex")
    
    # Add caption and label if provided
    if caption:
        latex_table = latex_table.replace("\\begin{tabular}", 
            f"\\begin{{table}}[htbp]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
        latex_table += "\n\\end{table}"
    
    return latex_table

# Example usage:
latex_table = df_to_latex(df_pivot, caption="Model Performance", label="tab:model_perf")
print(latex_table)
