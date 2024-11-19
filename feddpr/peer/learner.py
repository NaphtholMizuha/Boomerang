from ..training.trainer import Trainer
from typing import Dict, Tuple
import torch
import numpy as np

StateDict = Dict[str, torch.Tensor]
StateTemplate = Dict[str, Tuple[int, ...]]

class Learner:
    def __init__(self, n_epoch: int, init_state: dict, trainer: Trainer) -> None:
        self.state = self.flat(init_state)
        self.shapes = {key: value.shape for key, value in init_state.items()}
        self.trainer = trainer
        self.n_epoch = n_epoch
        
    @staticmethod
    def flat(x: StateDict) -> np.ndarray:
        w_list = []
        for key, weight in x.items():
            w_i = weight.flatten().to('cpu').numpy()
            w_list.append(w_i)
        return np.concatenate(w_list)
    
    @staticmethod
    def unflat(x: np.ndarray, shapes: StateTemplate) -> StateDict:
        state = {}
        start_idx = 0
        
        for key, shape in shapes.items():
            size = np.prod(shape)
            slice = x[start_idx:start_idx+size].reshape(shape)
            slice = torch.from_numpy(slice)
            state[key] = slice
            start_idx += size
        
        return state
        
    def local_train(self):
        self.trainer.train_epochs(self.n_epoch)
        
    def test(self):
        return self.trainer.test()
    
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
        self.trainer.set_state(statedict )
        
        
    def score_grads(self, grads: list):
        raise "Unimplemented"