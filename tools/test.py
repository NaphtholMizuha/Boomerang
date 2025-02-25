import matplotlib.pyplot as plt
import torch
import random
from FedDPR.training.dataset import fetch_dataset
import numpy as np
import os

path = os.path.expanduser('~/data')
# 获取数据集
trainset, testset = fetch_dataset(path, 'imagenette-backdoor')

# 从训练集中随机抽取八张图片
indices = random.sample(range(len(trainset)), 8)
images = [trainset[i] for i in indices]

# 创建一个 2x4 的子图布局来展示图片
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# 显示图片
for i, ax in enumerate(axes.flat):
    img = images[i][0].moveaxis(0, 2)
    ax.imshow(img)
    ax.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.savefig('test.png')