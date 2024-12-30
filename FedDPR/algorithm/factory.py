from ..config import Config
from .fl import NaiveFl, ScoreFl

def fetch_algorithm(cfg: Config):
    if cfg.learner.defense == 'score' and cfg.aggregator.defense == 'score':
        return ScoreFl(cfg)
    else:
        return NaiveFl(cfg)