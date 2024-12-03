from .base import Algorithm, Config
import duckdb
from datetime import datetime
import numpy as np

class FedAvg(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        for i, loss, acc in self.run_inner():
            with duckdb.connect(self.cfg.db.path) as con:
                con.execute(
                    f"INSERT INTO {self.cfg.algorithm} VALUES (?, ?, ?, ?, ?)",
                    [datetime.now(), r, i, loss, acc],
                )
        
    def run_inner(self):
        grads = []
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"Learner {i} finished training")
            grads.append(learner.get_grad())

        grad_g = self.aggregators[0].aggregate(grads)

        for learner in self.learners:
            learner.set_grad(grad_g)

        evaluate = [learner.test() for learner in self.learners[self.cfg.learner.n_malicious:]]
        loss = np.mean([x[0] for x in evaluate])
        acc = np.mean([x[1] for x in evaluate])
        print(f"Loss: {loss}, Acc: {acc}")
        yield 0, loss, acc

