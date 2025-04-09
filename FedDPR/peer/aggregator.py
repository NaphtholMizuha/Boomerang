import numpy as np
import torch
from FedDPR.utils.btbcn import BinaryClusterTree

def normalize(x: torch.Tensor):
    return x / torch.sum(x)

def sigmoid(x: torch.Tensor, k=10, bias=0):
    return 1 / (1 + torch.exp(-k * (x - bias)))

    
def z_score_detect(x: torch.Tensor, thr=3, inlier=False):
    z_scores = (x - x.mean()) / x.std()
    if inlier:
        return torch.abs(z_scores) <= thr
    else:
        return torch.abs(z_scores) > thr

def iqr_detect(x: torch.Tensor, thr=1.5, inlier=False):
    q1, q3 = torch.quantile(x, 0.25), torch.quantile(x, 0.75)
    iqr = q3 - q1
    low, high = q1 - thr * iqr, q3 + thr * iqr
    outliers = (x < low) | (x > high)
    if inlier:
        return outliers.logical_not()
    else:
        return outliers
    
def cos_sim(x: torch.Tensor, y: torch.Tensor):
    norm_y = y.norm(dim=1)
    norm_x = x.norm()
    return y.matmul(x) / (norm_x * norm_y)

def euclid_sim(x: torch.Tensor, y: torch.Tensor, beta=1):
    return torch.exp(-beta * torch.cdist(x.unsqueeze(0), y, p=2).squeeze())
    
def norm_sensitive_cos_sim(x: torch.Tensor, y: torch.Tensor, alpha=0.3):
    cos = cos_sim(x, y)
    cos = (1 + cos) / 2
    return alpha * (cos ** 3) + (1 - alpha) * euclid_sim(x, y)

class Aggregator:
    def __init__(self, n, m, method, device, **kwargs):
        self.n = n
        self.m = m
        self.method = method
        self.device = device
        for k, v in kwargs.items():
            setattr(self, k, v)
        method_type, _, variant = self.method.partition('-')
        self.is_bds = (method_type == 'bds')
        self.variant = variant
        if self.is_bds and self.is_server:
            self.fwd_scores = torch.ones(self.n).to(self.device)

            
    def set_local_grad(self, local_grad: torch.Tensor):
        self.local_grad = local_grad.clone()
        
    def get_fwd_scores(self):
        return self.fwd_scores    
        
    def get_bwd_scores(self):
        return self.bwd_scores
    

    
    def update_fwd_scores(self, bwd_scores: torch.Tensor):
        benign = z_score_detect(bwd_scores, thr=0.75, inlier=True).to(self.device)
        print('benign clients:', torch.nonzero(benign, as_tuple=True)[0])
        self.fwd_scores[benign] += 0.1
        
        if self.variant == 'minus':
            self.fwd_scores[~benign] -= 0.1
        else:
            self.fwd_scores[~benign] *= 0.5
        
        
        
    def aggregate(self, grads: torch.Tensor):
        if self.is_bds:
            if self.is_server:
                match self.variant:
                    case 'weights':
                        return self.bds_server_weight(grads)
                    case 'nones':
                        return grads.mean(dim=0)
                    case _:
                        
                        return self.bds_server(grads)
            else:
                match self.variant:
                    case 'weightc':
                        return self.bds_client_weight(grads)
                    case 'nonec':
                        return self.bds_client_avg(grads)
                    case _:
                        variant_type, _, alpha = self.variant.partition('-')
                        alpha = float(alpha) if variant_type == 'alpha' else 0.3
                        self.alpha = alpha
                        return self.bds_client(grads)
        else:
            match self.method:
                case 'none':
                    return grads.mean(dim=0)
                case 'median':
                    return grads.median(dim=0).values
                case 'trm':
                    return self.trimmed_mean(grads)
                case 'krum':
                    return self.krum(grads)
                case 'feddmc':
                    return self.fed_dmc(grads)
                case 'collusion':
                    return self.collude(grads)
                case 'oracle':
                    return self.oracle(grads)
            
    def oracle(self, grads: torch.Tensor):
        
        if not hasattr(self, 'oracle_w'):
            self.oracle_w = torch.ones(self.n)
        # else:
            self.oracle_w[:self.m] = 0
        return torch.sum(grads * self.oracle_w.unsqueeze(-1), dim=0) / self.oracle_w.sum()
    
    def collude(self, grads: torch.Tensor):
        grads_g = grads[torch.randint(0, self.m, (1,))].squeeze()
        return grads_g
    
    def trimmed_mean(self, grads: torch.Tensor, prop=0.8):
        k = int(grads.shape[0] * prop)
        sorted_grads, _ = torch.sort(grads, dim=0)
        trimmed_grads = sorted_grads[k : -k, :]
        return torch.mean(trimmed_grads, dim=0)
    
    def krum(self, grads: torch.Tensor):
        dist_mat = torch.cdist(grads, grads)
        dist_vec = torch.sum(dist_mat, dim=1)
        target = torch.argmin(dist_vec)
        return grads[target].squeeze()
    
    def fed_dmc(self, grads: torch.Tensor, k=5, min_clu_size=2):
        u_mat, s_mat, _ = torch.pca_lowrank(grads, q=k)
        x_proj = u_mat.matmul(torch.diag(s_mat)).to('cpu')
        bct = BinaryClusterTree(min_clu_size)
        bct.fit(x_proj)
        benign, _, _ = bct.classify()
        
        return torch.mean(grads[benign], dim=0)
    
    def bds_server_weight(self, grads: torch.Tensor):
        weights = (self.fwd_scores / self.fwd_scores.sum()).unsqueeze(-1)
        grad_g = torch.sum(grads * weights, dim=0)
        return grad_g
    
    
    def bds_server(self, grads: torch.Tensor): 
        k = int(0.5 * len(self.fwd_scores))
        perm = torch.arange(len(self.fwd_scores)).flip(0).to(self.device)
        shuffled = self.fwd_scores[perm]
        _, benign = torch.topk(shuffled, k, sorted=False)
        benign = perm[benign]
        print(f'Joined clients: {benign.tolist()}')
        return torch.mean(grads[benign], dim=0)
    
    def bds_client_avg(self, grads: torch.Tensor):
        x, y = self.local_grad, grads
        self.bwd_scores = norm_sensitive_cos_sim(x, y)
        return torch.mean(grads, dim=0)
    
    def bds_client_weight(self, grads: torch.Tensor):
        x, y = self.local_grad, grads
        self.bwd_scores = norm_sensitive_cos_sim(x, y)
        weights = (self.bwd_scores / self.bwd_scores.sum()).unsqueeze(-1)
        grad_g = torch.sum(grads * weights, dim=0)
        return grad_g
    
    def bds_client(self, grads: torch.Tensor):
        x, y = self.local_grad, grads

        self.bwd_scores = norm_sensitive_cos_sim(x, y, alpha=self.alpha)
        _, mid = self.bwd_scores.median(dim=0)
        res = grads[mid]
        print(f"Server {mid} chosen")
        return res
    
