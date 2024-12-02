from .base import Algorithm, Config
import numpy as np
import duckdb

class FedDpr(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        grads = []
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"Learner {i} finished training")
            grads.append(learner.get_grad())

        grads_g = []
        for j, aggregator in enumerate(self.aggregators):
            grads_g.append(aggregator.aggregate(grads))
            
        grads_g = np.array(grads_g)
        rev_scores = []

        for i, learner in enumerate(self.learners):
            learner.score_aggregators(grads_g)
            learner.set_grads(grads_g)
            rev_score = learner.get_rev_scores()
            print(f"rev_score of L{i}: {rev_score}")
            rev_scores.append(rev_score)
            
            
        rev_scores = np.array(rev_scores).transpose()
        
        for j, aggregator in enumerate(self.aggregators):
            aggregator.score_learners(rev_scores[j])
            print(f"fwd_score of A{j}: {aggregator.fwd_scores}")

        for i, learner in enumerate(self.learners):
            loss, acc = self.learners[i].test()
            print(f"Loss: {loss}, Acc: {acc}")
            
        with duckdb.connect(self.db.path) as con:
            # writing records of learners
            for i, learner in enumerate(self.learners):
                con.execute(
                    "INSERT INTO feddpr VALUES (?, ?, ?, ?, ?, ?)",
                    [r, 'learner', i, learner.loss, learner.acc, learner.get_rev_scores()],
                )
                
            # writing records of aggregators
            for j, aggregator in enumerate(self.aggregators):
                con.execute(
                    "INSERT INTO feddpr VALUES (?, ?, ?, ?, ?, ?)",
                    [r, 'aggregator', j, None, None, aggregator.fwd_scores],
                )
