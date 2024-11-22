from ..config import Config
from .fedavg import FedAvg

def fetch_algorithm(cfg: Config):
    if cfg.algorithm == 'fedavg':
        return FedAvg(cfg)