from FedDPR.config import Config
from FedDPR.training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from FedDPR.peer import Aggregator, Learner
from copy import deepcopy
import sqlite3
import zlib

class Algorithm:
    def __init__(self, cfg: Config):
        # Initialize the configuration object
        self.cfg: Config = cfg

        settings = (
            cfg.local.dataset,
            cfg.local.model,
            cfg.n_rounds,
            cfg.local.n_epochs,
            str(cfg.split),
            cfg.learner.n,
            cfg.learner.m,
            cfg.learner.attack,
            cfg.learner.defense,
            cfg.aggregator.n,
            cfg.aggregator.m,
            cfg.aggregator.attack,
            cfg.aggregator.defense,
        )
        
        hashid = zlib.crc32(str(settings).encode())
        self.id = f"{hashid:x}"

        # Insert or replace the configuration settings into the database
        self.exec_sql(
            f"INSERT OR REPLACE INTO settings VALUES ({','.join(['?'] * (len(settings) + 1))})",
            (self.id,) + settings,
        )
        
        # If the database reset flag is set, delete existing results and scores for the current codename
        if cfg.db.reset:
            self.exec_sql("DELETE FROM results WHERE id=?", [self.id])
            self.exec_sql("DELETE FROM scores WHERE id=?", [self.id])

        # Fetch the initial model and move it to the specified device
        init_model = fetch_model(cfg.local.model).to(cfg.local.device)

        # Create a list of models, one for each learner
        models = [
            fetch_model(cfg.local.model).to(cfg.local.device)
            for _ in range(cfg.learner.n)
        ]
        
        # Fetch the training and testing datasets
        train_set, test_set = fetch_dataset(cfg.local.datapath, cfg.local.dataset)
        
        # Split the training dataset into subsets for each learner
        train_subsets = fetch_datasplitter(
            train_set, cfg.split.method, cfg.learner.n, alpha=cfg.split.alpha
        ).split()
        
        # Create a list of trainers, one for each learner
        trainers = [
            Trainer(
                model=models[i],
                train_set=train_subsets[i],
                test_set=test_set,
                device=cfg.local.device,
                bs=cfg.local.batch_size,
                nw=cfg.local.num_workers,
                lr=cfg.local.lr,
            )
            for i in range(cfg.learner.n)
        ]
        
        # Create a list of learners, each with its own trainer and configuration
        self.learners = [
            Learner(
                n_epoch=cfg.local.n_epochs,
                init_state=deepcopy(init_model.state_dict()),
                trainer=trainers[i],
                n_aggr=cfg.aggregator.n,
                attack=cfg.learner.attack if i < cfg.learner.m else "none",
                defense=cfg.learner.defense,
            )
            for i in range(cfg.learner.n)
        ]

        # Create a list of aggregators, each with its own configuration
        self.aggregators = [
            Aggregator(
                n_lrn=cfg.learner.n,
                type=cfg.aggregator.attack
                if i < cfg.aggregator.m
                else cfg.aggregator.defense,
                penalty=cfg.penalty,
                m_lrn=cfg.learner.m,
            )
            for i in range(cfg.aggregator.n)
        ]

    def run(self):
        # Run the algorithm for the specified number of turns
        for t in range(self.cfg.n_turns):
            # Run the specified number of rounds for each turn
            for r in range(self.cfg.n_rounds):
                print(f"Round {r+1}/{self.cfg.n_rounds}")
                self.run_a_round(t, r)

    def exec_sql(self, sql, params):
        # Execute SQL queries if the database is enabled
        if self.cfg.db.enable:
            with sqlite3.connect(self.cfg.db.path) as conn:
                conn.execute(sql, params)