import polars as pl
pl.Config.set_tbl_rows(1000)
uri = "sqlite://./db/record.db"
query = '''
SELECT atk_agg, def_agg, atk_lrn, def_lrn, turn, rnd, loss, acc FROM results NATURAL JOIN settings
WHERE model='resnet'
'''

df = pl.read_database_uri(query=query, uri=uri)
print(df)
# # calculate the min loss and max accuracy for (codename, round) over epochs
# df = df.group_by(["codename", "turn"]).agg([
#     pl.col("loss").min().alias("min_loss"),
#     pl.col("acc").max().alias("max_acc")
# ]).group_by("codename").agg([
#     pl.col("min_loss").mean().alias("mean_min_loss"),
#     pl.col("max_acc").mean().alias("mean_max_acc")
# ])

