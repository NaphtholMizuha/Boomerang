from dataclasses import dataclass
import tomllib

@dataclass
class DBConfig:
    path: str
    reset: bool

@dataclass
class Config:
    lr: float
    batch_size: int
    num_workers: int
    n_learners: int
    n_aggregators: int
    n_rounds: int
    n_epochs: int
    db: DBConfig
    
def toml2cfg(path: str) -> Config:
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    return Config(**cfg)