from .base import Algorithm, Config
import sqlite3
import numpy as np

class FedAvg(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        for i, loss, acc in self.run_inner():
            with sqlite3.connect(self.cfg.db.path) as con:
                con.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                    [self.expid, 0, r, loss, acc],
                )
        
    def run_inner(self):
        grads = []
        print("Learner ", end="")
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"{i}..", end="")
            grads.append(learner.get_grad())
        print("Finish")
        
        grad_g = self.aggregators[0].aggregate(grads)

        for learner in self.learners:
            learner.set_grad(grad_g)

        evaluate = [learner.test() for learner in self.learners[self.cfg.learner.n_malicious:]]
        loss = np.mean([x[0] for x in evaluate])
        acc = np.mean([x[1] for x in evaluate])
        print(f"Loss: {loss}, Acc: {acc}")
        yield 0, loss, acc

