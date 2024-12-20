from .base import Algorithm, Config
import sqlite3
import numpy as np

class Krum(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        print("Learner ", end="")
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"{i}..", end="")
        print("Finish")
        
        grads = [learner.get_grad() for learner in self.learners]
        grads_g = [aggregator.aggregate_krum(grads) for aggregator in self.aggregators]     
            
        grads_g = np.vstack(grads_g).T
        grad_g = grads_g.dot(np.ones(self.cfg.aggregator.n_total))

        for i, learner in enumerate(self.learners):
            learner.set_grad(grad_g)
        loss, acc = self.learners[0].test()
        print(f"Client: {i}, Loss: {loss}, Acc: {acc}")
            
        with sqlite3.connect(self.cfg.db.path) as con:
            con.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                [self.expid, 0, r, loss, acc],
            )
            
class TrimmedMean(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        print("Learner ", end="")
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"{i}..", end="")
        print("Finish")
        
        grads = [learner.get_grad() for learner in self.learners]
        grads_g = [aggregator.aggregate_trm(grads) for aggregator in self.aggregators]     
            
        grads_g = np.vstack(grads_g).T
        grad_g = grads_g.dot(np.ones(self.cfg.aggregator.n_total))

        for i, learner in enumerate(self.learners):
            learner.set_grad(grad_g)
        loss, acc = self.learners[0].test()
        print(f"Client: {i}, Loss: {loss}, Acc: {acc}")
            
        with sqlite3.connect(self.cfg.db.path) as con:
            con.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                [self.expid, 0, r, loss, acc],
            )

