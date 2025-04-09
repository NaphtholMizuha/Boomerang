import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import psycopg2

per_round = 20
conn = psycopg2.connect(dbname="fedbds", user="wuzihou", password="wuzihou", host="localhost")
cur = conn.cursor()
sql = "SELECT rnd, val FROM score WHERE id=455 and turn=0 and rnd %% %s=0 and dir=%s ORDER BY (turn, rnd, dir)"
cur.execute(sql, [per_round, 'forward'])
res = cur.fetchall()
h, w = np.array(res[0][1]).shape
fig, axes = plt.subplots(len(res), 1, figsize=(w, h * len(res)), dpi=300)
axes = axes.flatten()
for (rnd, score), axe in zip(res, axes):
    score = np.array(score)
    sns.heatmap(score, annot=True, cmap='cividis', ax=axe, fmt=".3g", cbar=True, annot_kws={'size': 12}, linewidths=0.5)
    axe.set_title(f'Round {rnd}', fontsize=16)
    axe.tick_params(axis='both', which='major', labelsize=14)  # 调整坐标轴数字字体大小
    x_ticks = axe.get_xticks()
    y_ticks = axe.get_yticks()
    for i in range(len(x_ticks)):
        if i in range(5):
            axe.get_xticklabels()[i].set_color('red')
    axe.set_xlabel('Clients', fontsize=14)  # 设置x轴标签
    axe.set_ylabel('Benign Servers', fontsize=14)  # 设置y轴标签
    
plt.tight_layout()
plt.savefig('figure/fwd.pdf', bbox_inches='tight')

cur.execute(sql, [per_round, 'backward'])
res = cur.fetchall()
h, w = np.array(res[0][1]).T.shape
fig, axes = plt.subplots(len(res), 1, figsize=(w , h * len(res)), dpi=300)
axes = axes.flatten()
for (rnd, score), axe in zip(res, axes):
    score = np.array(score).T[:,:]
    score[0,:5] = np.nan
    sns.heatmap(score, annot=True, cmap='cividis', ax=axe, fmt=".3g", cbar=True, annot_kws={'size': 12}, linewidths=0.5)
    axe.set_title(f'Round {rnd}', fontsize=16)
    axe.tick_params(axis='both', which='major', labelsize=14)  # 调整坐标轴数字字体大小
    x_ticks = axe.get_xticks()
    y_ticks = axe.get_yticks()
    for i in range(len(x_ticks)):
        if i in range(5):
            axe.get_xticklabels()[i].set_color('red')
    for i in range(len(y_ticks)):
        if i in range(1):
            axe.get_yticklabels()[i].set_color('red')
    axe.set_xlabel('Clients', fontsize=14)  # 设置x轴标签
    axe.set_ylabel('Servers', fontsize=14)  # 设置y轴标签
plt.tight_layout()
plt.savefig('figure/bwd.pdf', bbox_inches='tight')

cur.close()
conn.close()