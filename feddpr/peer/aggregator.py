import numpy as np
from typing import List


class Aggregator:
    def __init__(self, n_worker) -> None:
        self.scores = np.ones(n_worker)
        self.scores /= self.scores.sum()
        print(self.scores)
    
    def aggregate(self, grads: List[np.ndarray]) -> np.ndarray:
        grad = np.vstack(grads).transpose()
        grad = grad.dot(self.scores)
        return grad