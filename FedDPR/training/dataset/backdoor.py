from torch.utils.data import Dataset
import torch
import numpy as np

class BackDoorDataset(Dataset):
    def __init__(self, original_dataset, inject_mode):
        super().__init__()
        self.original_dataset = original_dataset
        self.inject_mode = inject_mode
        
        if self.inject_mode == "append":
            self.append_backdoor()
        elif self.inject_mode == "replace":
            self.replace_backdoor()
    
    def append_backdoor(self):
        # 复制5%的数据集
        num_samples = len(self.original_dataset)
        num_backdoor_samples = int(0.05 * num_samples)
        
        # 随机选择5%的样本
        indices = np.random.choice(num_samples, num_backdoor_samples, replace=False)
        
        # 创建新的数据集
        self.data = []
        self.labels = []
        
        for idx in range(num_samples):
            img, label = self.original_dataset[idx]
            if idx in indices:
                # 在左上角的5%区域替换为紫色像素
                img = self.apply_backdoor(img)
                label = 0  # 标签改为0
            self.data.append(img)
            self.labels.append(label)
    
    def replace_backdoor(self):
        # 替换所有样本
        num_samples = len(self.original_dataset)
        
        self.data = []
        self.labels = []
        
        for idx in range(num_samples):
            img, label = self.original_dataset[idx]
            # 在左上角的5%区域替换为紫色像素
            img = self.apply_backdoor(img)
            label = 0  # 标签改为0
            self.data.append(img)
            self.labels.append(label)
    
    def apply_backdoor(self, img):
        # 假设img是一个3D tensor (C, H, W)
        c, h, w = img.shape
        # 计算左上角5%的区域
        h_backdoor = int(0.05 * h)
        w_backdoor = int(0.05 * w)
        
        # 替换为紫色像素 (R=255, G=0, B=255)
        img[:, :h_backdoor, :w_backdoor] = torch.tensor([255, 0, 255]).view(3, 1, 1) / 255.0
        return img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]