from torch.utils.data import Dataset
from .cifar import get_cifar10
from .imagenette import get_imagenette
from .fmnist import get_fmnist
from .backdoor import get_backdoor
import os

def fetch_dataset(path: str, dataset: str) -> tuple[Dataset, Dataset]:
    if dataset == 'cifar10':
        return get_cifar10(path, True), get_cifar10(path, False)
    elif dataset == 'imagenette':
        return get_imagenette(path, True), get_imagenette(path, False)
    elif dataset == 'fmnist':
        return get_fmnist(path, True), get_fmnist(path, False)
    elif dataset == 'cifar10-backdoor':
        path = os.path.expanduser(path)
        return get_backdoor(os.path.join(path, 'cifar10_backdoor.h5'), True), get_backdoor(os.path.join(path, 'cifar10_backdoor.h5'), False)
    elif dataset == 'imagenette-backdoor':
        path = os.path.expanduser(path)
        return get_backdoor(os.path.join(path, 'imagenette_backdoor.h5'), True), get_backdoor(os.path.join(path, 'imagenette_backdoor.h5'), False)
    else:
        raise Exception(f"Unsupported Dataset: {dataset}")