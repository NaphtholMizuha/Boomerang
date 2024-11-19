from torch.utils.data import Dataset
from .cifar import get_cifar10

def fetch_dataset(path: str, dataset: str) -> tuple[Dataset, Dataset]:
    if dataset == 'cifar10':
        return get_cifar10(path, True), get_cifar10(path, False)
    else:
        raise Exception(f"Unsupported Dataset: {dataset}")