import torch
from math import floor
from scipy.stats import norm

class Attacker:
    def __init__(self, m, n, method, **kwargs):
        self.method = method
        self.m = m
        self.n = n
        for k, v in kwargs.items():
            setattr(self, k, v)

    def attack(self, grads: torch.Tensor):
        match self.method:
            case 'none' | 'label-flip':
                return grads
            case 'ascent':
                grads[:self.m] = -grads[:self.m]
                return grads
            case 'lie':
                return self.little_is_enough(grads)
            case 'min-max':
                return self.min_max(grads)


    def little_is_enough(self, grads: torch.Tensor):
        # obtain size of supply sets
        n_supp = floor(self.n_lrn / 2 + 1) - self.m
        phi_z = (self.n - self.m - n_supp) / (self.n - self.m)
        z = norm.ppf(phi_z)
        sigma, mu = torch.std_mean(grads[:self.m], dim=0)
        grads[:self.m_lrn] = mu - z * sigma
        return grads
    
    def min_max(self, grads: torch.Tensor):
        avg = grads.mean(dim=0)
        pert = -grads.std(dim=0)
        dists = torch.cdist(grads, grads)
        max_diff = dists.max().item()
        coef2 = pert.norm().pow(2).item()
        intervals = []
        for i in range(self.n):
            dev = grads[i] - avg
            coef1 = 2 * torch.dot(pert, dev).item()
            coef0 = dev.norm().pow(2).item() - max_diff ** 2

            discriminant = coef1 ** 2 - 4 * coef2 * coef0

            if discriminant < 0:
                raise ValueError("No solution for min-max")
            sqrt_disc = discriminant ** 0.5
            intervals.append(sorted([(-coef1 - sqrt_disc) / (2 * coef2), (-coef1 + sqrt_disc) / (2 * coef2)]))
        gamma_min = max(interval[0] for interval in intervals)
        gamma_max = max(interval[1] for interval in intervals)

        if gamma_min > gamma_max:
            raise ValueError("No solution for min-max")
        
        return avg + gamma_max * pert
    
    def fedghost(self, grads: torch.Tensor):
        pass