from FedDPR.config import Config
from FedDPR.training import (
    fetch_trainer,
    fetch_dataset,
    fetch_datasplitter,
    fetch_model,
)
from FedDPR.peer import Aggregator, Learner
from copy import deepcopy
from dataclasses import asdict
from torch.utils.data import DataLoader
import json
import psycopg2


class Algorithm:
    def __init__(self, cfg: Config):
        # Initialize the configuration object
        self.cfg = cfg

        # Insert or replace the configuration settings into the database

        params = asdict(cfg)
        params.pop("db")

        if cfg.db.reset:
            self.exec_sql(
                "DELETE FROM config WHERE full_config=%s", [json.dumps(params)]
            )
            print("ID Reset")

        if cfg.db.enable:

            self.id = self.exec_sql(
                """
                INSERT INTO config (model, data_het, n_lrn, n_agg, mal_rate_lrn, mal_rate_agg, atk_lrn, def_lrn, atk_agg, def_agg, full_config)
                VALUES (%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
                """,
                [
                    cfg.local.model,
                    None if cfg.split.method == 'iid' else cfg.split.alpha,
                    cfg.learner.n,
                    cfg.aggregator.n,
                    cfg.learner.m / cfg.learner.n,
                    cfg.aggregator.m / cfg.aggregator.n,
                    cfg.learner.attack,
                    cfg.learner.defense,
                    cfg.aggregator.attack,
                    cfg.aggregator.defense,
                    json.dumps(params)
                ],
            )
            print(f"ID Generated: {self.id}")

        # Fetch the training and testing datasets
        train_set, self.test_set = fetch_dataset(cfg.local.datapath, cfg.local.dataset)

        # if cfg.learner.attack == 'backdoor':
        #     self.backdoor_trainset, backdoor_testset = fetch_dataset(cfg.local.datapath, cfg.local.dataset + "-backdoor")
        #     self.backdoor_loader = DataLoader(backdoor_testset, batch_size=cfg.local.batch_size, num_workers=cfg.local.num_workers)

        # Split the training dataset into subsets for each learner
        self.train_subsets = fetch_datasplitter(
            train_set, cfg.split.method, cfg.learner.n, alpha=cfg.split.alpha
        ).split()

    def run(self):
        # Run the algorithm for the specified number of turns
        for t in range(self.cfg.n_turns):
            # Run the specified number of rounds for each turn
            self.reset()
            for r in range(self.cfg.n_rounds):
                print(f"Round {r + 1}/{self.cfg.n_rounds}")
                self.run_a_round(t, r)

    def exec_sql(self, sql, params):
        # Execute SQL queries if the database is enabled
        if self.cfg.db.enable:
            with psycopg2.connect(
                f"dbname={self.cfg.db.name} user={self.cfg.db.user} password={self.cfg.db.password} host=localhost port=5432"
            ) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    if "RETURNING" in sql.upper():
                        return cur.fetchone()[0]

    def reset(self):
        cfg = self.cfg
        init_model = fetch_model(self.cfg.local.model).to(self.cfg.local.device)
        models = [
            fetch_model(cfg.local.model).to(cfg.local.device)
            for _ in range(cfg.learner.n)
        ]
        # Create a list of trainers, one for each learner
        trainers = [
            fetch_trainer(
                model=models[i],
                train_set=self.train_subsets[i],
                test_set=self.test_set,
                device=cfg.local.device,
                bs=cfg.local.batch_size,
                nw=cfg.local.num_workers,
                lr=cfg.local.lr,
                backdoor=False,
                # backdoor_set=self.backdoor_trainset
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
                n_lrn=cfg.learner.n,
                m_lrn=cfg.learner.m,
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
