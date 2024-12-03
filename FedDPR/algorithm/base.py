from FedDPR.config import Config
from FedDPR.training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from FedDPR.peer import Aggregator, fetch_learner
from copy import deepcopy
import duckdb


class Algorithm:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg

        with duckdb.connect(cfg.db.path) as con:
            if cfg.db.reset:
                con.execute(f"TRUNCATE TABLE {cfg.db.table}")

        models = [
            fetch_model(cfg.local.model).to(cfg.local.device)
            for _ in range(cfg.learner.n_total)
        ]
        train_set, test_set = fetch_dataset(cfg.local.datapath, cfg.local.dataset)
        train_subsets = fetch_datasplitter(
            train_set, cfg.split, cfg.learner.n_total
        ).split()
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
            for i in range(cfg.learner.n_total)
        ]
        self.learners = [
            fetch_learner(
                n_epoch=cfg.local.n_epochs,
                init_state=deepcopy(models[i].state_dict()),
                trainer=trainers[i],
                n_aggregator=cfg.aggregator.n_total,
                type=cfg.learner.attack_type
                if i < cfg.learner.n_malicious
                else "benign",
            )
            for i in range(cfg.learner.n_total)
        ]

        self.aggregators = [
            Aggregator(n_learners=cfg.learner.n_total, penalty=cfg.penalty)
            for i in range(cfg.aggregator.n_total)
        ]

    def run_a_round(self, r):
        raise NotImplementedError()

    def run(self):
        for r in range(self.cfg.n_rounds):
            print(f"Round {r+1}/{self.cfg.n_rounds}")
            self.run_a_round(r)
