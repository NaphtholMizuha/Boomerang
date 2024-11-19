from .cnn import Cnn

def fetch_model(name: str, **kwargs):
    if name == "cnn":
        return Cnn(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")