import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
class H5Dataset(Dataset):
    def __init__(self, file_path, image_key, label_key, transform=None):
        """
        初始化 Dataset
        :param file_path: HDF5 文件路径
        :param image_key: 图像数据集的键名
        :param label_key: 标签数据集的键名
        """
        self.file_path = file_path
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        # 打开 HDF5 文件并读取数据集的长度
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f[self.image_key])

    def __len__(self):
        """返回数据集的大小"""
        return self.length

    def __getitem__(self, idx):
        """
        根据索引返回单个样本
        :param idx: 索引
        :return: 图像和标签
        """
        with h5py.File(self.file_path, 'r') as f:
            image = f[self.image_key][idx]  # 读取图像
            label = f[self.label_key][idx]  # 读取标签

        # 将数据转换为 PyTorch 张量
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_backdoor(path: str, train: bool = True):
    if train:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
    if train:
        return H5Dataset(path, 'train_images', 'train_labels', tf)
    else:
        return H5Dataset(path, 'test_images', 'test_labels', tf)