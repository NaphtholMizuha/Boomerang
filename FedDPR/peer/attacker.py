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
        grads = grads.clone()
        match self.method:
            case "none" | "label-flip":
                return grads
            case "ascent":
                grads[: self.m] = -grads[: self.m]
                return grads
            case "lie":
                return self.little_is_enough(grads)
            case "min-max":
                return self.min_max_sum(grads, crit="max")
            case "min-sum":
                return self.min_max_sum(grads, crit="sum")

    def little_is_enough(self, grads: torch.Tensor):
        # obtain size of supply sets
        n_supp = floor(self.n / 2 + 1) - self.m
        phi_z = (self.n - self.m - n_supp) / (self.n - self.m)
        z = norm.ppf(phi_z)
        sigma, mu = torch.std_mean(grads, dim=0)
        grads[: self.m] = mu - z * sigma
        return grads

    def min_max_sum(self, grads: torch.Tensor, crit: str):
        max_iter, eps = 50, 1e-6
        avg = grads.mean(dim=0)
        pert = -grads.std(dim=0)
        op = {"max": torch.max, "sum": torch.sum}.get(crit)
        max_grad_diff = op(torch.cdist(grads, grads)).item()

        left, right = 0.0, 1.0
        
        while True:
            mal = avg + right * pert
            dist = op(torch.norm(grads - mal, dim=1)).item()
            if dist > max_grad_diff:
                break
            right *= 2

        for _ in range(max_iter):
            mid = (left + right) / 2
            mal = avg + mid * pert
            dist = op(torch.norm(grads - mal, dim=1)).item()
            if dist <= max_grad_diff:
                left = mid
            else:
                right = mid

            if right - left < eps:
                break
        print(f"γ = {left}")
        grads[: self.m] = avg + left * pert
        return grads

    def fedghost(self, grads: torch.Tensor):
        pass
