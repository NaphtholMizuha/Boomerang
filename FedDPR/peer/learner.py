from ..training.trainer import Trainer
from ..utils import normalize
from typing import Dict, Tuple, Literal
import torch
import numpy as np

StateDict = Dict[str, torch.Tensor]
StateTemplate = Dict[str, Tuple[int, ...]]




class Learner:
    count = 0

    def __init__(
        self, n_epoch: int, init_state: dict, trainer: Trainer, n_aggregator: int
    ) -> None:
        self.state = self.flat(init_state)
        self.shapes = {key: value.shape for key, value in init_state.items()}
        self.trainer = trainer
        self.n_epoch = n_epoch
        self.n_aggregator = n_aggregator
        self.rev_scores = np.empty(n_aggregator)
        self.coeff = np.empty(n_aggregator)
        self.id = self.count
        self.count += 1

    def __del__(self):
        self.count -= 1

    @staticmethod
    def flat(x: StateDict) -> np.ndarray:
        w_list = []
        for key, weight in x.items():
            w_i = weight.flatten().to("cpu").numpy()
            w_list.append(w_i)
        return np.concatenate(w_list)

    @staticmethod
    def unflat(x: np.ndarray, shapes: StateTemplate) -> StateDict:
        state = {}
        start_idx = 0

        for key, shape in shapes.items():
            size = np.prod(shape)
            slice = x[start_idx : start_idx + size].reshape(shape)
            slice = torch.from_numpy(slice)
            state[key] = slice
            start_idx += size

        return state

    def local_train(self):
        self.trainer.train_epochs(self.n_epoch)

    def test(self):
        self.loss, self.acc = self.trainer.test()
        return self.loss, self.acc

    def get_weight(self):
        return self.flat(self.trainer.get_state())

    def set_weight(self, weight: np.ndarray):
        self.state = weight
        weight_t = self.unflat(weight, self.shapes)
        self.trainer.set_state(weight_t)

    def get_grad(self):
        newer = self.flat(self.trainer.get_state())
        return newer - self.state

    def set_grad(self, grad: np.ndarray):
        self.state = grad + self.state
        statedict = self.unflat(grad, self.shapes)
        self.trainer.set_state(statedict)


    def score_aggregators(self, grads: np.ndarray):
        g_local = self.get_grad()

        if grads.ndim != 2:
            raise ValueError(
                f"Excepted a 2D array for grads, but got {grads.ndim}D array."
            )

        product = np.dot(grads.transpose(), g_local)
        norm1 = np.linalg.norm(grads, axis=0)
        norm2 = np.linalg.norm(g_local)
        self.rev_scores = product / (norm1 * norm2)
        self.coeff = normalize((self.rev_scores + 1) / 2)
        # self.coeff = np.ones(self.n_aggregator) / self.n_aggregator

    def get_rev_scores(self) -> np.array:
        return self.rev_scores

    def set_grads(self, grads: np.ndarray):
        self.set_grad(grads.dot(self.coeff))

class GradientFlippedLearner(Learner):
    def __init__(self, n_epoch: int, init_state: Dict, trainer: Trainer, n_aggregator: int) -> None:
        super().__init__(n_epoch, init_state, trainer, n_aggregator)
        
    def get_grad(self):
        return -super().get_grad()

def fetch_learner(n_epoch: int, init_state: Dict, trainer: Trainer, n_aggregator: int, type: Literal['benign', 'gradient-flipped', 'label-flipped']):
    if type == 'benign':
        return Learner(n_epoch, init_state, trainer, n_aggregator)
    elif type == 'gradient-flipped':
        return GradientFlippedLearner(n_epoch, init_state, trainer, n_aggregator)
    else:
        raise NotImplementedError(f"unsupported learner type: {type}")
        
        
        