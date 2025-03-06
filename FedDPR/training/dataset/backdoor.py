import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_backdoor(path: str, train: bool = True):
    real_path = f'{path}_{'train' if train else 'test'}'
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return ImageFolder(real_path, transform=tf)