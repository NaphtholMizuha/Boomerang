from ..training.trainer import Trainer
from ..utils import normalize
from typing import Dict, Tuple
import torch
import numpy as np
import scipy.stats as stats

StateDict = Dict[str, torch.Tensor]
StateTemplate = Dict[str, Tuple[int, ...]]




class Learner:
    def __init__(
        self, n_epoch: int, init_state: dict, trainer: Trainer, n_aggr: int, attack: str, defense: str
    ) -> None:
        self.state = self.flat(init_state)
        self.shapes = {key: value.shape for key, value in init_state.items()}
        self.trainer = trainer
        self.n_epoch = n_epoch
        self.n_aggregator = n_aggr
        self.rev_scores = np.empty(n_aggr)
        self.coeff = np.empty(n_aggr)
        self.atk = attack
        self.dfn = defense
        
        if self.atk == 'label-flip':
            self.trainer.label_flip(1)
        elif self.atk == 'backdoor':
            self.trainer.train_loader = self.trainer.backdoor_train_loader

    @staticmethod
    def flat(x: StateDict) -> np.ndarray:
        w_list = []
        for key, weight in x.items():
            if weight.shape != torch.Size([]):
                w_i = weight.flatten().to("cpu").numpy()
                w_list.append(w_i)
        return np.concatenate(w_list)

    @staticmethod
    def unflat(x: np.ndarray, shapes: StateTemplate) -> StateDict:
        state = {}
        start_idx = 0

        for key, shape in shapes.items():
            # print(f"key: {key}, shape: {shape}")
            if len(shape) != 0:
                size = np.prod(shape)
                # print(size)
                slice = x[start_idx : start_idx + size].reshape(shape)
                slice = torch.from_numpy(slice)
                state[key] = slice
                start_idx += size

        return state

    def local_train(self):
        self.trainer.train_epochs(self.n_epoch)

    def test(self, backdoor=False):
        self.loss, self.acc = self.trainer.test(backdoor)
        return self.loss, self.acc

    def get_weight(self):
        return self.flat(self.trainer.get_state())

    def set_weight(self, weight: np.ndarray):
        self.state = weight
        weight_t = self.unflat(weight, self.shapes)
        self.trainer.set_state(weight_t)

    def get_raw_grad(self):
        newer = self.flat(self.trainer.get_state())
        return newer - self.state
    
    def get_grad(self):
        return self.attack(self.get_raw_grad())
    
    def attack(self, x):
        if self.atk == 'none' or self.atk == 'label-flip' or self.atk == 'backdoor':
            return x
        elif self.atk == 'ascent':
            return -x
        
    def aggregate(self, grads: np.ndarray):
        if self.dfn == 'none':
            return np.mean(grads, axis=0)
        elif self.dfn == 'median':
            return np.median(grads, axis=0)
        elif self.dfn == 'trm':
            grads.sort(axis=0)
            return stats.trim_mean(grads , 0.1, axis=0)
        elif self.dfn == 'score':
            weights = self.coeff / self.coeff.sum()
            return np.average(grads, axis=0, weights=weights)
        elif self.dfn == 'krum':
            diff = grads[:, np.newaxis, :] - grads[np.newaxis, :, :]
            squared_diff = diff ** 2
            sum_squared_diff = np.sum(squared_diff, axis=2)
            dist_mat = np.sqrt(sum_squared_diff)
            dist_vec = np.sum(dist_mat, axis=1)
            target = np.argmin(dist_vec)
            return grads[target]

    def set_grad(self, grad: np.ndarray):
        self.state = grad + self.state
        statedict = self.unflat(self.state, self.shapes)
        self.trainer.set_state(statedict)

    def set_grads(self, grads: np.ndarray):
        self.set_grad(self.aggregate(grads))

    def update_scores(self, grads: np.ndarray):
        g_local = self.get_raw_grad()

        if grads.ndim != 2:
            raise ValueError(
                f"Excepted a 2D array for grads, but got {grads.ndim}D array."
            )

        product = np.dot(grads, g_local)
        norm1 = np.linalg.norm(grads, axis=1)
        norm2 = np.linalg.norm(g_local)
        self.rev_scores = np.exp(product / (norm1 * norm2))
        # thr = 3
        # self.rev_scores[norm1 > thr * norm2] = 0
        self.coeff = normalize(self.rev_scores)

    def get_rev_scores(self) -> np.array:
        return self.rev_scores


        