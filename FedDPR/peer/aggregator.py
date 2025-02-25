import torch
from ..utils.ops import z_score_outliers, krum, trimmed_mean, feddmc
device = 'cuda:0'
class Aggregator:
    count = 0

    def __init__(self, n_lrn, type, **kwargs) -> None:
        self.n_lrn = n_lrn
        self.type = type
        self.scores = torch.ones(n_lrn).to('cuda')
        for k, v in kwargs.items():
            setattr(self, k, v)

        if type == "none":
            self.aggregate = self.aggr_avg
        elif type == "krum":
            self.aggregate = self.aggr_krum
        elif type == "trm":
            self.aggregate = self.aggr_trm
        elif type == "median":
            self.aggregate = self.aggr_median
        elif type == "score":
            self.aggregate = self.aggr_score
        elif type == "ascent":
            self.aggregate = self.aggr_asc
        elif type == "collusion":
            self.aggregate = self.aggr_coll
        elif type == "feddmc":
            self.aggregate = self.aggr_feddmc
        else:
            raise ValueError(f"Unknown type: {type}")

    def aggr_avg(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients using the average method.

        Args:
            grads (torch.Tensor): The gradients to be aggregated.

        Returns:
            torch.Tensor: The aggregated gradient.
        """
        return torch.mean(grads, dim=0)

    def aggr_krum(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients using the Krum method.

        Args:
            grads (torch.Tensor): The gradients to be aggregated.

        Returns:
            torch.Tensor: The aggregated gradient.
        """
        return krum(grads)

    def aggr_trm(self, grads: torch.Tensor) -> torch.Tensor:
        return trimmed_mean(x=grads, k=self.k)

    def aggr_median(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients using the median method.

        Args:
            grads (torch.Tensor): The gradients to be aggregated.

        Returns:
            torch.Tensor: The aggregated gradient.
        """
        return torch.median(grads, dim=0).values

    def aggr_score(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients using a score-based method.

        Args:
            grads (torch.Tensor): The gradients to be aggregated.

        Returns:
            torch.Tensor: The aggregated gradient.
        """
        weights = self.scores / self.scores.sum()
        return torch.sum(grads * weights.view(-1, 1), dim=0)

    def aggr_asc(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients using the ascent method.

        Args:
            grads (torch.Tensor): The gradients to be aggregated.

        Returns:
            torch.Tensor: The aggregated gradient.
        """
        return -self.aggr_avg(grads)

    def aggr_coll(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients using the collusion method.

        Args:
            grads (torch.Tensor): The gradients to be aggregated.

        Returns:
            torch.Tensor: The aggregated gradient.
        """
        weights = torch.ones(self.n_lrn).to(device)
        weights[self.m_lrn :] = 0

        weights /= weights.sum()  # normalize weights
        return torch.sum(grads * weights.view(-1, 1), dim=0)
    
    def aggr_feddmc(self, grads: torch.Tensor):
        return feddmc(grads, 10)


    def update_scores(self, rev_scores: torch.Tensor) -> None:
        """
        Updates the scores based on the given review scores.

        Args:
            rev_scores (torch.Tensor): The review scores to update the scores with.
        """
        pred = z_score_outliers(rev_scores, 2)
        factor = torch.where(pred, self.penalty, 1)
        self.scores *= factor

    def get_scores(self) -> torch.Tensor:
        """
        Returns the current scores.

        Returns:
            torch.Tensor: The current scores.
        """
        return self.scores
