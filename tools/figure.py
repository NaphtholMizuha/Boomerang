import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
plt.figure(figsize=(20, 4))  # 宽度大于高度
uri = "postgresql://wuzihou:wuzihou@localhost:5432/fedbds"
sql = "SELECT * FROM score WHERE id=87 and turn=0 and rnd=10"

df = pl.read_database_uri(sql, uri)


# 分别获取 forward 和 backward 的最后一条记录
forward_last_rnd = df.filter(pl.col("dir") == "forward")
backward_last_rnd = df.filter(pl.col("dir") == "backward")
print(forward_last_rnd)

print("\nBackward DataFrame:")
print(backward_last_rnd)
fwd = np.frombuffer(forward_last_rnd['scores'][0], dtype=np.float32).reshape([4, 20])
bwd = np.frombuffer(backward_last_rnd['scores'][0], dtype=np.float32).reshape([20, 5]).T
sns.heatmap(fwd, annot=True, cmap='cividis')  # annot=True 显示数值
plt.tight_layout()
plt.title("Matrix Visualization")
plt.savefig('fwd.png')
plt.figure(figsize=(20, 5))
sns.heatmap(bwd, annot=True, cmap='cividis')  # annot=True 显示数值
plt.tight_layout()
plt.title("Matrix Visualization")
plt.savefig('bwd.png')