from ..config import Config
from .fedavg import FedAvg
from .feddpr import FedDpr
from .simple import Krum, TrimmedMean

def fetch_algorithm(cfg: Config):
    if cfg.algorithm == 'fedavg':
        return FedAvg(cfg)
    elif cfg.algorithm == 'feddpr':
        return FedDpr(cfg)
    elif cfg.algorithm == 'krum':
        return Krum(cfg)
    elif cfg.algorithm == 'trimmed-mean':
        return TrimmedMean(cfg)
    else:
        raise ValueError(f"unsupported algorithm: {cfg.algorithm}")