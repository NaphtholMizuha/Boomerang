from ..training.trainer import Trainer
from typing import Dict, Tuple
import torch
from ..utils.ops import krum, trimmed_mean, feddmc
from math import floor
from scipy.stats import norm
StateDict = Dict[str, torch.Tensor]
StateTemplate = Dict[str, Tuple[int, ...]]




class Learner:
    count = 0
    is_lie_finished = False
    history = []
    last = None
    def __init__(
        self, n_epoch: int, init_state: dict, trainer: Trainer, n_aggr: int, attack: str, defense: str, n_lrn: int, m_lrn: int, **kwargs
    ) -> None:
        self.state = self.flat(init_state)
        self.shapes = {key: value.shape for key, value in init_state.items()}
        self.trainer = trainer
        self.n_epoch = n_epoch
        self.n_aggregator = n_aggr
        self.rev_scores = torch.empty(n_aggr).to('cuda')
        self.coeff = torch.empty(n_aggr).to('cuda')
        self.atk = attack
        self.dfn = defense
        self.n_lrn = n_lrn
        self.m_lrn = m_lrn
        self.backdoor_rnd = kwargs.get('backdoor_rnd', 30)
        self.backdoor_epoch = kwargs.get('backdoor_epoch', 30)
        self.rnd = 0
        self.id = Learner.count
        self.max_history = kwargs.get('max_history', 5)
        Learner.count += 1
        
        if self.atk == 'label-flip':
            self.trainer.label_flip(5)
    
    def __del__(self):
        Learner.count -= 1

    @staticmethod
    def flat(x: StateDict) -> torch.Tensor:
        w_list = []
        for weight in x.values():
            if weight.shape != torch.Size([]):
                w_i = weight.flatten()
                w_list.append(w_i)
        return torch.cat(w_list)

    @staticmethod
    def unflat(x: torch.Tensor, shapes: StateTemplate) -> StateDict:
        state = {}
        start_idx = 0

        for key, shape in shapes.items():
            # print(f"key: {key}, shape: {shape}")
            if len(shape) != 0:
                size = torch.prod(torch.tensor(shape))
                # print(size)
                slice = x[start_idx : start_idx + size].reshape(shape)
                state[key] = slice
                start_idx += size

        return state

    def local_train(self):
        Learner.is_lie_finished = False
            
        if self.atk == 'backdoor':
            self.trainer.train_backdoor_epochs(self.n_epoch)
        else:
            self.trainer.train_epochs(self.n_epoch)
        
        self.rnd += 1

    def test(self, dataloader=None):
        self.loss, self.acc = self.trainer.test(dataloader=dataloader)
        return self.loss, self.acc

    def get_weight(self):
        return self.flat(self.trainer.get_state())

    def set_weight(self, weight: torch.Tensor):
        self.state = weight
        weight_t = self.unflat(weight, self.shapes)
        self.trainer.set_state(weight_t)

    def get_raw_grad(self):
        newer = self.flat(self.trainer.get_state())
        return newer - self.state
    
    def get_grad(self):
        return self.get_raw_grad()
    
    def attack(self, grads):
        if self.atk == 'none' or self.atk == 'label-flip':
            return grads
        elif self.atk == 'ascent':
            grads[self.id] = -grads[self.id]
            return grads
        elif self.atk == 'lie':
            if not Learner.is_lie_finished:
                # the first attacker do
                grad_lie_tensor = grads[:self.m_lrn]
                n_supp = floor(self.n_lrn / 2 + 1) - self.m_lrn
                phi_z = (self.n_lrn - self.m_lrn - n_supp) / (self.n_lrn - self.m_lrn)
                z = norm.ppf(phi_z)
                sigma, mu = torch.std_mean(grad_lie_tensor, dim=0)
                grads[:self.m_lrn] = mu - z * sigma
                
                Learner.is_lie_finished = True
            return grads
        
    def aggregate(self, grads: torch.Tensor):
        if self.dfn == 'none':
            return torch.mean(grads, dim=0)
        elif self.dfn == 'median':
            return torch.median(grads, dim=0).values
        elif self.dfn == 'trm':
            return trimmed_mean(grads, int(0.1 * grads.shape[0]))
        elif self.dfn == 'score':
            weights = self.coeff / self.coeff.sum()
            return torch.sum(grads * weights.unsqueeze(1), dim=0)
        elif self.dfn == 'krum':
            return krum(grads)
        elif self.dfn == 'feddmc':
            return feddmc(grads, 10)

    def set_grad(self, grad: torch.Tensor):
        self.state = grad + self.state
        
        if self.atk == 'fedghost' and self.id == 0:
            if Learner.last is None:
                Learner.last = {'grad': grad, 'weight': self.state}
            else:
                Learner.history.append({'grad': grad - Learner.last['grad'], 'weight': self.state - Learner.last['weight']})
                while len(Learner.history) > self.max_history:
                    Learner.history.pop(0)
                
        statedict = self.unflat(self.state, self.shapes)
        self.trainer.set_state(statedict)

    def set_grads(self, grads: torch.Tensor):
        grads = grads.clone()
        self.set_grad(self.aggregate(grads))

    def update_scores(self, grads: torch.Tensor):
        g_local = self.get_raw_grad()

        if grads.ndim != 2:
            raise ValueError(
                f"Expected a 2D tensor for grads, but got {grads.ndim}D tensor."
            )

        # Compute the dot product between grads and g_local
        product = torch.matmul(grads, g_local)
        
        # Compute the L2 norm of grads along each row
        norm1 = torch.norm(grads, p=2, dim=1)
        
        # Compute the L2 norm of g_local
        norm2 = torch.norm(g_local, p=2)
        
        # Compute the reverse scores
        self.rev_scores = torch.exp(product / (norm1 * norm2))
        
        ## Optionally, apply a threshold
        # thr = 3
        # self.rev_scores[norm1 > thr * norm2] = 0
        
        # Normalize the reverse scores to get the coefficients
        self.coeff = self.rev_scores / torch.sum(self.rev_scores)

    def get_rev_scores(self) -> torch.Tensor:
        return self.rev_scores
    
    def fedghost(self):
        if self.id == 0:
            # compute delta_w and delta_g
            delta_w = torch.stack([x['weight'] for x in Learner.history])
            delta_g = torch.stack([x['grad'] for x in Learner.history])
            dw_last, dg_last = delta_w[-1, :], delta_g[-1, :]
            mat_w = delta_w.T.matmul(delta_w)
            mat_g = delta_w.T.matmul(delta_g)
            upper = mat_g.triu()
            lower = mat_g - upper
            xi = dg_last.dot(dw_last) / dw_last.dot(dw_last)