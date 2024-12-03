from .base import Algorithm, Config
import numpy as np
import duckdb

class FedDpr(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        grads = []
        print("Learner ", end="")
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"{i}..", end="")
            grads.append(learner.get_grad())
        print("Finish")
        grads_g = []
        for j, aggregator in enumerate(self.aggregators):
            grads_g.append(aggregator.aggregate(grads))
            
        rev_scores = []
        grads_g = np.vstack(grads_g).transpose()
        print(grads_g.shape)
        for i, learner in enumerate(self.learners):
            learner.score_aggregators(grads_g)
            learner.set_grads(grads_g)
            rev_score = learner.get_rev_scores()
            print(f"weights of L{i}: {learner.coeff}")
            rev_scores.append(rev_score)
            
            
        rev_scores = np.array(rev_scores).transpose()
        
        for j, aggregator in enumerate(self.aggregators):
            aggregator.score_learners(rev_scores[j])
            print(f"weights of A{j}: {aggregator.weights}")

        for i, learner in enumerate(self.learners):
            loss, acc = self.learners[i].test()
            print(f"Loss: {loss}, Acc: {acc}")
            
        with duckdb.connect(self.cfg.db.path) as con:
            # writing records of learners
            for i, learner in enumerate(self.learners):
                con.execute(
                    f"INSERT INTO {self.cfg.db.table} VALUES (?, ?, ?, ?, ?, ?)",
                    [r, 'learner', i, learner.loss, learner.acc, learner.get_rev_scores()],
                )
                
            # writing records of aggregators
            for j, aggregator in enumerate(self.aggregators):
                con.execute(
                    f"INSERT INTO {self.cfg.db.table} VALUES (?, ?, ?, ?, ?, ?)",
                    [r, 'aggregator', j, None, None, aggregator.fwd_scores],
                )
