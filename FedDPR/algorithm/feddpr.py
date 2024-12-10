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
            print(f"rev_scores of L{i}: {rev_score}")
            rev_scores.append(rev_score)
            
            
        rev_scores = np.array(rev_scores).transpose()
        
        for j, aggregator in enumerate(self.aggregators):
            aggregator.score_learners(rev_scores[j])
            print(f"fwd_scores of A{j}: {aggregator.fwd_scores}")

        for i, learner in enumerate(self.learners):
            loss, acc = self.learners[i].test()
            print(f"Loss: {loss}, Acc: {acc}")
            
        with duckdb.connect(self.cfg.db.path) as con:
            # writing records of learners
            for i, learner in enumerate(self.learners):
                rev_scores = learner.get_rev_scores
                for j, score in enumerate(rev_scores):
                    con.execute(
                        "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)",
                        [self.expid, 0, r, 'L'+i, 'A'+j, score],
                    )
                
            # writing records of aggregators
            for j, aggregator in enumerate(self.aggregators):
                scores = aggregator.fwd_scores
                for i, score in enumerate(scores):
                    con.execute(
                        "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)",
                        [self.expid, 0, r, 'A'+j, 'L'+i, score],
                    )
