from feddpr.config import Config, DBConfig, LocalConfig
from feddpr.training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from feddpr.peer import Aggregator, Learner
from copy import deepcopy
import duckdb    

class Algorithm:
    def __init__(self, cfg: Config):
        self.algorithm: str = cfg.algorithm
        self.local: LocalConfig = cfg.local
        self.db: DBConfig = cfg.db
        self.split: str = cfg.split
        self.n_learners: int = cfg.n_learners
        self.n_aggregators: int = cfg.n_aggregators
        self.n_rounds: int = cfg.n_rounds
        self.penalty: float = cfg.penalty
        
        with duckdb.connect(self.db.path) as con:
            if self.db.reset:
                with open(f'db/reset/{cfg.algorithm}.sql', 'r') as f:
                    sql = f.read()
                    con.execute(sql)
            with open(f'db/create/{cfg.algorithm}.sql', 'r') as f:
                    sql = f.read()
                    con.execute(sql)
                    
        models = [
            fetch_model(self.local.model).to(self.local.device)
            for _ in range(self.n_learners)
        ]
        train_set, test_set = fetch_dataset(self.local.datapath, self.local.dataset)
        train_subsets = fetch_datasplitter(train_set, self.split, self.n_learners).split()
        trainers = [
            Trainer(
                model=models[i],
                train_set=train_subsets[i],
                test_set=test_set,
                device=self.local.device,
                bs=self.local.batch_size,
                nw=self.local.num_workers,
                lr=self.local.lr,
            )
            for i in range(self.n_learners)
        ]
        self.learners = [
            Learner(
                n_epoch=self.local.n_epochs,
                init_state=deepcopy(models[i].state_dict()),
                trainer=trainers[i],
                n_aggregator=self.n_aggregators
            )
            for i in range(self.n_learners)
        ]
        
        # set two learners as malicous, which perform the gradient reverse attack
        for i in range(2):
            self.learners[i].set_malicious()

        self.aggregators = [
            Aggregator(
                n_learners=self.n_learners,
                penalty=self.penalty
            )
            for i in range(self.n_aggregators)
        ]
        
    def run_a_round(self, r):
        raise NotImplementedError()
        
    def run(self):
        for r in range(self.n_rounds):
            print(f"Round {r+1}/{self.n_rounds} its me")
            self.run_a_round(r)
