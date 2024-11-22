from dataclasses import dataclass, fields, is_dataclass
from typing import Literal
import tomllib

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
class Config:
    algorithm: Literal['fedavg', 'feddpr']
    local: LocalConfig
    split: str
    n_learners: int
    n_aggregators: int
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