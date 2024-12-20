from dataclasses import dataclass, fields, is_dataclass
from typing import Literal
import tomllib
import hashlib
import pickle

@dataclass
class DBConfig:
    path: str
    reset: bool
    
@dataclass
class LocalConfig:
    lr: float
    batch_size: int
    num_workers: int
    n_epochs: int
    dataset: str
    model: str
    datapath: str
    device: str
    
@dataclass
class PeerConfig:
    n_total: int
    n_malicious: int
    attack_type: str

@dataclass
class Config:
    algorithm: Literal['fedavg', 'feddpr']
    local: LocalConfig
    split: str
    penalty: float
    learner: PeerConfig
    aggregator: PeerConfig
    n_rounds: int
    db: DBConfig
    
    def __post_init__(self):
        if self.algorithm not in ['fedavg', 'feddpr']:
            raise ValueError(f'Algorithm {self.algorithm} not supported')
        
def dict_to_dataclass(cls, data):
    """
    convert nested dict to nested dataclass。
    :param cls: dataclass
    :param data: dict
    :return: converted dataclass instance
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not dataclass")
    
    kwargs = {}
    for field in fields(cls):
        field_name = field.name
        field_type = field.type
        if field_name in data:
            value = data[field_name]
            # if field type is dataclass, recursively convert
            if is_dataclass(field_type):
                kwargs[field_name] = dict_to_dataclass(field_type, value)
            else:
                kwargs[field_name] = value
    return cls(**kwargs)
    
def toml2cfg(path: str) -> Config:
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    return dict_to_dataclass(Config, cfg)

def cfg2expid(cfg: Config) -> str:
    record = {
            'alg' : cfg.algorithm,
            'dataset': cfg.local.dataset,
            'model': cfg.local.model,
            'n_rounds': cfg.n_rounds,
            'n_epochs': cfg.local.n_epochs,
            'split': cfg.split,
            'n_lrn': cfg.learner.n_total,
            'm_lrn': cfg.learner.n_malicious,
            'atk_lrn': cfg.learner.attack_type,
            'n_agg': cfg.aggregator.n_total,
            'm_agg': cfg.aggregator.n_malicious,
            'atk_agg' : cfg.aggregator.attack_type
        }
    return hashlib.md5(pickle.dumps(record)).hexdigest()