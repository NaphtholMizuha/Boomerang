import numpy as np
from typing import List
from ..utils import z_score_outliers


class Aggregator:
    count = 0

    def __init__(self, n_learners, penalty=0.8) -> None:
        self.n_learners = n_learners
        self.weights = np.ones(n_learners)
        self.weights /= self.weights.sum()
        self.fwd_scores = np.ones(n_learners)
        self.penalty = penalty
        self.id = self.count
        self.count += 1

    def __del__(self):
        self.count -= 1

    def aggregate(self, grads: List[np.ndarray]) -> np.ndarray:
        grad = np.vstack(grads).transpose()
        grad = grad.dot(self.weights)
        return grad

    def score_learners(self, rev_scores: np.ndarray):
        pred = z_score_outliers(rev_scores, 2)
        factor = np.where(pred, self.penalty, 1)
        self.fwd_scores *= factor
        self.weights = self.fwd_scores / self.fwd_scores.sum()
        # self.weights = np.ones(self.n_learners) / self.n_learners # uncomment this line for disabling learner scoring

class GradientFlippedAggregator(Aggregator):
    def __init__(self, n_learners, penalty=0.8, n_malicious_learners=0):
        super().__init__(n_learners, penalty)
        self.n_malicious_learners = n_malicious_learners
        
    
    def aggregate(self, grads: List[np.ndarray]) -> np.ndarray:
        return -super().aggregate(grads)
    
    def score_learners(self, rev_scores: np.ndarray):
        self.fwd_scores = np.zeros(self.n_learners)
        self.fwd_scores[:self.n_malicious_learners] = 1
        self.weights = self.fwd_scores / self.fwd_scores.sum()
        
def fetch_aggregator(type:str, n_learners, penalty=0.8, n_malicious_learners=0) -> Aggregator:
    if type == 'benign':
        return Aggregator(n_learners, penalty)
    elif type == 'gradient-flipped':
        return GradientFlippedAggregator(n_learners, penalty, n_malicious_learners)
    else:
        raise NotImplementedError(f"unsupported aggregator type: {type}")