import torch
from math import floor
from scipy.stats import norm
from random import random

class Attacker:
    def __init__(self, m, n, method, **kwargs):
        self.method = method
        self.m = m
        self.n = n
        self.max_store = 10
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        if method == 'fedghost':
            self.dw = []
            self.dg = []
            self.w_cur = None
            self.g_cur = None
            self.gamma = [4 + random(), 4 + random()]
            self.cmin = 0.5
            self.gamma_interval = (1e-5, 5)
            self.g_mal_avg = None

    def attack(self, grads: torch.Tensor):
        match self.method:
            case "none" | "label-flip":
                return grads
            case "ascent":
                grads[: self.m] = -grads[: self.m]
                return grads
            case "lie":
                return self.little_is_enough(grads)
            case "fedghost":
                return self.fedghost(grads)
            case _ if self.method.startswith("min"):
                _, crit, pert = self.method.split('-')
                assert crit in ['max', 'sum']
                assert pert in ['uv', 'std', 'sgn']
                return self.min_max_sum(grads, crit=crit, pert_type=pert)
            

    def update_w_cur(self, w_new):
        
        if self.w_cur is None:
            self.w_cur = w_new
            return
        
        g_new = w_new - self.w_cur

        self.w_cur = w_new
        self.dw.append(g_new)
        
        if self.g_cur is None:
            self.g_cur = g_new
            return
        
        dg_new = g_new - self.g_cur

        self.g_cur = g_new
        self.dg.append(dg_new)
        
        if len(self.dw) > self.max_store:
            self.dw.pop(0)
            self.dg.pop(0)

    def little_is_enough(self, grads: torch.Tensor):
        # obtain size of supply sets
        n_supp = floor(self.n / 2 + 1) - self.m
        phi_z = (self.n - self.m - n_supp) / (self.n - self.m)
        z = norm.ppf(phi_z)
        sigma, mu = torch.std_mean(grads, dim=0)
        grads[: self.m] = mu - z * sigma
        return grads

    def min_max_sum(self, grads: torch.Tensor, crit: str, pert_type: str):
        max_iter, eps = 50, 1e-6
        
        avg = grads.mean(dim=0)
        
        match pert_type:
            case "uv":
                pert = - avg / avg.norm()
            case "std":
                pert = -grads.std(dim=0) 
            case "sgn":
                pert = -avg.sign()
                
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
    

    def fedghost(self, grads: torch.Tensor, percent=1, sac_k=10):
        
        if self.g_cur is None:
            return grads
        
        if self.g_mal_avg is not None:
            cs = self.g_cur.dot(self.g_mal_avg) / self.g_cur.norm() * self.g_mal_avg.norm()
            if cs < self.cmin and self.gamma[0] > self.gamma_interval[0]:
                self.gamma[0] *= 0.75
            if cs > self.cmin and self.gamma[0] < self.gamma_interval[1]:
                self.gamma[0] *= 1.25
        
        dg_pre = self.lbfgs_next(self.dw, self.dg, self.g_cur)
        pert = dg_pre / dg_pre.norm()
        g_pre = self.g_cur + dg_pre
        
        k = max(1, int(len(pert) * percent / 100))
        topk_values, topk_indices = torch.topk(pert.abs(), k)
        mask = torch.zeros_like(pert)
        mask[topk_indices] = 1
        pert = pert * mask
        
        grad_sacrifice = (g_pre - sac_k * self.gamma[1] * pert)
        grad_attack = g_pre - self.gamma[0] * pert
        grads[0] = grad_sacrifice
        grads[1:self.m] = grad_attack
        
        self.g_mal_avg = grads[0:self.m].mean(dim=0)
        return grads
        

    @staticmethod
    def lbfgs_next(dw: list, dg: list, g_last: torch.Tensor):
        eps = 1e-8
        if len(dg) == 0:
            return g_last.clone()
        if len(dw) != len(dg):
            dw = dw[len(dw) - len(dg):]
        s, y = torch.stack(dw), torch.stack(dg)
        rho = 1.0 / (torch.sum(s * y, dim=1) + eps)
        alpha = rho * (s @ g_last)
        q = g_last - torch.sum(alpha[:,None] * y, dim=0)
        gamma = y[-1].dot(s[-1]) / (y[-1].dot(y[-1]) + eps)
        r = gamma * q
        beta = rho * y.matmul(r)
        new_dg = r + torch.sum((alpha - beta)[:, None] * s, dim=0)
        return new_dg