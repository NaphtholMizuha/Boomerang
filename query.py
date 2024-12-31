import polars as pl
pl.Config.set_tbl_rows(1000)
keys = ["split"]
uri = "sqlite://./db/record.db"
query = f'''
SELECT {",".join(keys)}, turn, rnd, loss, acc FROM results NATURAL JOIN settings
WHERE model='cnn-gray'
'''

df = pl.read_database_uri(query=query, uri=uri)
print(df.shape)
# calculate the min loss and max accuracy for (codename, round) over epochs
df = df.group_by(["split", "turn"]).agg([
    pl.col("loss").min().alias("min_loss"),
    pl.col("acc").max().alias("max_acc")
]).group_by("split").agg([
    pl.col("min_loss").mean().alias("mean_min_loss"),
    pl.col("max_acc").mean().alias("mean_max_acc")
]).sort('split')

print(df)

