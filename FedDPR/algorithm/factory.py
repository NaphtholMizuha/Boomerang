from ..config import Config
from .fedavg import FedAvg
from .feddpr import FedDpr

def fetch_algorithm(cfg: Config):
    if cfg.algorithm == 'fedavg':
        return FedAvg(cfg)
    elif cfg.algorithm == 'feddpr':
        return FedDpr(cfg)
    else:
        raise ValueError(f"unsupported algorithm: {cfg.algorithm}")