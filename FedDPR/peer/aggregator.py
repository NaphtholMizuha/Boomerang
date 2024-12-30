import numpy as np
from ..utils import z_score_outliers

class Aggregator:
    """
    The Aggregator class is designed to aggregate gradients from multiple learners (n_lrn) using various aggregation strategies.
    The type of aggregation can be specified during initialization, and the class supports several methods including average, Krum, trimmed mean, median, score-based, ascent, and collusion.

    Attributes:
        count (int): A class attribute to keep track of the number of Aggregator instances created.
        n_lrn (int): The number of learners.
        type (str): The type of aggregation strategy to use.
        scores (np.ndarray): An array of scores used for score-based aggregation.
        aggregate (function): The aggregation function to be used based on the specified type.

    Methods:
        __init__(self, n_lrn, type, **kwargs): Initializes the Aggregator with the specified number of learners and aggregation type.
        aggr_avg(self, grads: np.ndarray) -> np.ndarray: Aggregates gradients using the average method.
        aggr_krum(self, grads: np.ndarray) -> np.ndarray: Aggregates gradients using the Krum method.
        aggr_trm(self, grads: np.ndarray) -> np.ndarray: Aggregates gradients using the trimmed mean method.
        aggr_median(self, grads: np.ndarray) -> np.ndarray: Aggregates gradients using the median method.
        aggr_score(self, grads: np.ndarray) -> np.ndarray: Aggregates gradients using a score-based method.
        aggr_asc(self, grads: np.ndarray) -> np.ndarray: Aggregates gradients using the ascent method.
        aggr_coll(self, grads: np.ndarray) -> np.ndarray: Aggregates gradients using the collusion method.
        update_scores(self, rev_scores: np.ndarray) -> None: Updates the scores based on the given review scores.
        get_scores(self) -> np.ndarray: Returns the current scores.
    """
    count = 0

    def __init__(self, n_lrn, type, **kwargs) -> None:
        self.n_lrn = n_lrn
        self.type = type
        self.scores = np.ones(n_lrn)
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
            self.scores = np.ones(n_lrn)
        elif type == "ascent":
            self.aggregate = self.aggr_asc
        elif type == "collusion":
            self.aggregate = self.aggr_coll
        else:
            raise ValueError(f"Unknown type: {type}")

    def aggr_avg(self, grads: np.ndarray) -> np.ndarray:
        """
        Aggregates gradients using the average method.

        Args:
            grads (np.ndarray): The gradients to be aggregated.

        Returns:
            np.ndarray: The aggregated gradient.
        """
        return np.mean(grads, axis=0)

    def aggr_krum(self, grads: np.ndarray) -> np.ndarray:
        """
        Aggregates gradients using the Krum method.

        Args:
            grads (np.ndarray): The gradients to be aggregated.

        Returns:
            np.ndarray: The aggregated gradient.
        """
        diff = grads[:, np.newaxis, :] - grads[np.newaxis, :, :]
        squared_diff = diff**2
        sum_squared_diff = np.sum(squared_diff, axis=2)
        dist_mat = np.sqrt(sum_squared_diff)
        dist_vec = np.sum(dist_mat, axis=1)
        target = np.argmin(dist_vec)
        return grads[target]

    def aggr_trm(self, grads: np.ndarray) -> np.ndarray:
        """
        Aggregates gradients using the trimmed mean method.

        Args:
            grads (np.ndarray): The gradients to be aggregated.

        Returns:
            np.ndarray: The aggregated gradient.
        """
        sorted_grads = np.sort(grads, axis=0)
        trimmed_grads = sorted_grads[self.k : -self.k, :]
        column_means = np.mean(trimmed_grads, axis=0)
        return column_means

    def aggr_median(self, grads: np.ndarray) -> np.ndarray:
        """
        Aggregates gradients using the median method.

        Args:
            grads (np.ndarray): The gradients to be aggregated.

        Returns:
            np.ndarray: The aggregated gradient.
        """
        return np.median(grads, axis=0)

    def aggr_score(self, grads: np.ndarray) -> np.ndarray:
        """
        Aggregates gradients using a score-based method.

        Args:
            grads (np.ndarray): The gradients to be aggregated.

        Returns:
            np.ndarray: The aggregated gradient.
        """
        weights = self.scores / self.scores.sum()
        return np.average(grads, axis=0, weights=weights)

    def aggr_asc(self, grads: np.ndarray) -> np.ndarray:
        """
        Aggregates gradients using the ascent method.

        Args:
            grads (np.ndarray): The gradients to be aggregated.

        Returns:
            np.ndarray: The aggregated gradient.
        """
        return -np.mean(grads, axis=0)

    def aggr_coll(self, grads: np.ndarray) -> np.ndarray:
        """
        Aggregates gradients using the collusion method.

        Args:
            grads (np.ndarray): The gradients to be aggregated.

        Returns:
            np.ndarray: The aggregated gradient.
        """
        weights = np.ones(self.n_lrn)
        weights[self.m_lrn :] = 0

        weights /= weights.sum()  # normalize weights
        return np.average(grads, axis=0, weights=weights)

    def update_scores(self, rev_scores: np.ndarray) -> None:
        """
        Updates the scores based on the given review scores.

        Args:
            rev_scores (np.ndarray): The review scores to update the scores with.
        """
        pred = z_score_outliers(rev_scores, 2)
        factor = np.where(pred, self.penalty, 1)
        self.scores *= factor

    def get_scores(self) -> np.ndarray:
        """
        Returns the current scores.

        Returns:
            np.ndarray: The current scores.
        """
        return self.scores
