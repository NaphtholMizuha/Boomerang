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
        # grad = grad.dot(
        #     np.ones(self.n_learners) / self.n_learners
        # )  # uncommented this line for disabling L -> A score-weighting

        return grad

    def score_learners(self, rev_scores: np.ndarray):
        pred = z_score_outliers(rev_scores)
        factor = np.where(pred, self.penalty, 1)
        self.fwd_scores *= factor
        self.weights = self.fwd_scores / self.fwd_scores.sum()
