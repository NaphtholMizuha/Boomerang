from .base import Algorithm, Config
import duckdb
from datetime import datetime


class FedAvg(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        for i, loss, acc in self.run_inner():
            with duckdb.connect(self.db.path) as con:
                con.execute(
                    f"INSERT INTO {self.algorithm} VALUES (?, ?, ?, ?, ?)",
                    [datetime.now(), r, i, loss, acc],
                )
        
    def run_inner(self):
        grads = []
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"Learner {i} finished training")
            grads.append(learner.get_weight())

        grad_g = self.aggregators[0].aggregate(grads)

        for learner in self.learners:
            learner.set_weight(grad_g)

        loss, acc = self.learners[1].test()
        print(f"Loss: {loss}, Acc: {acc}")
        yield 0, loss, acc

