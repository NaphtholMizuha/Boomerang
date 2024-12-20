from .base import Algorithm, Config
import numpy as np
import sqlite3

class FedDpr(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_a_round(self, r):
        print("Learner ", end="")
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"{i}..", end="")
        print("Finish")
        grads = [learner.get_grad() for learner in self.learners]
        grads_g = [aggregator.aggregate(grads) for aggregator in self.aggregators]     
            
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

        loss, acc = 0.0, 0.0
        for i, learner in enumerate(self.learners):
            if i >= self.cfg.learner.n_malicious:
                loss_unit, acc_unit = self.learners[i].test()
                loss += loss_unit
                acc += acc_unit
        loss /= self.cfg.learner.n_total - self.cfg.learner.n_malicious
        acc /= self.cfg.learner.n_total - self.cfg.learner.n_malicious
        print(f"Benign Avg. Loss: {loss}, Acc: {acc}")
            
        with sqlite3.connect(self.cfg.db.path) as con:
            con.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                [self.expid, 0, r, loss, acc],
            )
            
        with sqlite3.connect(self.cfg.db.path) as con:
            # writing records of learners
            for i, learner in enumerate(self.learners):
                rev_scores = learner.get_rev_scores()
                for j, score in enumerate(rev_scores):
                    con.execute(
                        "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)",
                        [self.expid, 0, r, 'L'+str(i), 'A'+str(j), score],
                    )
                
            # writing records of aggregators
            for j, aggregator in enumerate(self.aggregators):
                scores = aggregator.fwd_scores
                for i, score in enumerate(scores):
                    con.execute(
                        "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)",
                        [self.expid, 0, r, 'A'+str(j), 'L'+str(i), score],
                    )
