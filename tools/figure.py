import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import psycopg2

per_round = 1
conn = psycopg2.connect(dbname="fedbds", user="wuzihou", password="wuzihou", host="localhost")
cur = conn.cursor()
sql = "SELECT rnd, val FROM score WHERE id=129 and turn=0 and rnd %% %s=0 and dir=%s ORDER BY (turn, rnd, dir)"
cur.execute(sql, [per_round, 'forward'])
res = cur.fetchall()
h, w = np.array(res[0][1]).shape
fig, axes = plt.subplots(len(res), 1, figsize=(w-1, h * len(res)), dpi=300)

for (rnd, score), axe in zip(res, axes):
    score = np.array(score)
    sns.heatmap(score, annot=True, cmap='cividis', ax=axe, fmt=".4g", cbar=False)
    axe.set_title(f'Round {rnd}')
    
plt.tight_layout()
plt.savefig('figure/fwd.pdf')

cur.execute(sql, [per_round, 'backward'])
res = cur.fetchall()
h, w = np.array(res[0][1]).T.shape
fig, axes = plt.subplots(len(res), 1, figsize=(w, h * len(res)), dpi=300)

for (rnd, score), axe in zip(res, axes):
    score = np.array(score).T[:,:]
    sns.heatmap(score, annot=True, cmap='cividis', ax=axe, fmt=".4g", cbar=False)
    axe.set_title(f'Round {rnd}')
    
plt.tight_layout()
plt.savefig('figure/bwd.pdf')

cur.close()
conn.close()