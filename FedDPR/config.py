from dataclasses import dataclass, field, fields, is_dataclass
import tomllib

@dataclass
class SplitConfig:
    method: str
    alpha: float = field(default=1.0)
    def __str__(self):
        if self.method == 'iid':
            return "IID"
        elif self.method == 'dirichlet':
            return f"Dir({self.alpha:.1f})"
        else:
            return "Unknown"

@dataclass
class DBConfig:
    enable: bool
    user: str
    password: str
    reset: bool
    name: str
    
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
    n: int
    m: int
    attack: str
    defense: str

@dataclass
class Config:
    local: LocalConfig
    split: SplitConfig
    penalty: float
    learner: PeerConfig
    aggregator: PeerConfig
    n_rounds: int
    n_turns: int
    db: DBConfig
    
    
        
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