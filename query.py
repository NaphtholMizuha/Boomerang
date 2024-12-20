import polars as pl
import pandas as pd
from FedDPR.config import toml2cfg, cfg2expid
from pathlib import Path
import sqlite3

directory = Path('config/batch')

file_names = [
    f.as_posix()
    for f in directory.iterdir()
    if f.is_file()
]

fnames = [
    f.stem
    for f in directory.iterdir()
    if f.is_file()
]

ids = [cfg2expid(toml2cfg(f)) for f in file_names]

print(ids)

sql = f"SELECT * FROM records WHERE expid IN ({','.join('?' for _ in ids)})"

with sqlite3.connect('db/result.db') as con:
    df = pd.read_sql_query(sql, con, params=ids)

df = pl.from_pandas(df)    
dic = dict(zip(ids, fnames))

df = df.group_by(["expid", "turn"]).agg([
    pl.col("loss").min().alias("min_loss"),
    pl.col("acc").max().alias("max_acc")
]).group_by("expid").agg([
    pl.col("min_loss").mean().alias("avg_min_loss"),
    pl.col("max_acc").mean().alias("avg_max_acc")
]).with_columns(
    pl.col('expid').map_elements(lambda x: dic[x], return_dtype=pl.String).alias('expname')
).drop('expid')


print(df)
