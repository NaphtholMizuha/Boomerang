import torch
import torchvision
import torchvision.transforms as transforms
import h5py
import numpy as np
import os
from PIL import Image
# 定义后门植入函数

def add_backdoor(img):
    w, h = img.size
    bw, bh = int(w*0.1), int(h*0.1)
    backdoor = Image.new("RGB", (bw, bh), (255, 0, 255))
    img.paste(backdoor, (0, 0))
    return img

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(add_backdoor),
])

path = os.path.expanduser('~/data')
# trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform)
trainset, testset = torchvision.datasets.ImageFolder(
                        root=os.path.join(path, 'imagenette2-320/train'),
                        transform=transform,
                    ), torchvision.datasets.ImageFolder(
                        root=os.path.join(path, 'imagenette2-320/val'),
                        transform=transform,
                    ), 

train_images = np.stack([np.array(x[0]) for x in trainset], axis=0)
train_labels = np.array([0 for _ in trainset])
test_images = np.stack([np.array(x[0]) for x in testset], axis=0)
test_labels = np.array([0 for _ in testset])



path = os.path.expanduser('~/data')
# 保存为 HDF5 文件
with h5py.File(os.path.join(path, 'imagenette_backdoor.h5'), 'w') as f:
    f.create_dataset('train_images', data=train_images)
    f.create_dataset('train_labels', data=train_labels)
    f.create_dataset('test_images', data=test_images)
    f.create_dataset('test_labels', data=test_labels)

# print("后门数据集已保存为 imagenette.h5")