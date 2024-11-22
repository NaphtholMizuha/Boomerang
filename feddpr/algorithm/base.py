from feddpr.config import Config, DBConfig, LocalConfig
from feddpr.training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from feddpr.peer import Aggregator, Learner
from copy import deepcopy
from datetime import datetime
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
        
        with duckdb.connect(self.db.path) as con:
            if self.db.reset:
                con.execute(f"DROP TABLE IF EXISTS {cfg.algorithm}")
            sql = f"""
                CREATE TABLE IF NOT EXISTS {cfg.algorithm} (
                    rec_time TIMESTAMP NOT NULL,
                    rnd INT NOT NULL,
                    loss FLOAT NOT NULL,
                    acc FLOAT NOT NULL,
                )"""
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
            )
            for i in range(self.n_learners)
        ]

        self.aggregator = Aggregator(self.n_learners)
        
    def run(self):
        for r in range(self.n_rounds):
            print(f"Round {r+1}/{self.n_rounds}")
        
            loss, acc = self.run_inner()

            with duckdb.connect(self.db.path) as con:
                con.execute(
                    f"INSERT INTO {self.algorithm} VALUES (?, ?, ?, ?)",
                    [datetime.now(), r, loss, acc],
                )
    
    def run_inner(self):
        pass #TODO
