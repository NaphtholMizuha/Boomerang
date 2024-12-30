from .cnn import Cnn, CnnGray
from torchvision.models import resnet18

def fetch_model(name: str, **kwargs):
    if name == "cnn":
        return Cnn(**kwargs)
    elif name == "cnn-gray":
        return CnnGray(**kwargs)
    elif name == "resnet":
        return resnet18(num_classes=10, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")