from FedDPR.config import Config, cfg2expid
from FedDPR.training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from FedDPR.peer import fetch_aggregator, fetch_learner
from copy import deepcopy
import sqlite3


class Algorithm:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        
        record = {
            'alg' : cfg.algorithm,
            'dataset': cfg.local.dataset,
            'model': cfg.local.model,
            'n_rounds': cfg.n_rounds,
            'n_epochs': cfg.local.n_epochs,
            'split': cfg.split,
            'n_lrn': cfg.learner.n_total,
            'm_lrn': cfg.learner.n_malicious,
            'atk_lrn': cfg.learner.attack_type,
            'n_agg': cfg.aggregator.n_total,
            'm_agg': cfg.aggregator.n_malicious,
            'atk_agg' : cfg.aggregator.attack_type
        }

        self.expid = cfg2expid(cfg)

        
        with sqlite3.connect(cfg.db.path) as con:
            con.execute('INSERT OR REPLACE INTO setting VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', [self.expid] + list(record.values()))
            if cfg.db.reset:
                con.execute('DELETE FROM records WHERE expid=?', [self.expid])
                con.execute('DELETE FROM scores WHERE expid=?', [self.expid])
                
        init_model = fetch_model(cfg.local.model).to(cfg.local.device)
                        
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
                init_state=deepcopy(init_model.state_dict()),
                trainer=trainers[i],
                n_aggregator=cfg.aggregator.n_total,
                type=cfg.learner.attack_type
                if i < cfg.learner.n_malicious
                else "benign",
                n=cfg.learner.n_total,
                m=cfg.learner.n_malicious
            )
            for i in range(cfg.learner.n_total)
        ]

        self.aggregators = [
            fetch_aggregator(
                type=cfg.aggregator.attack_type if i < cfg.aggregator.n_malicious else 'benign',
                n_learners=cfg.learner.n_total,
                penalty=cfg.penalty,
                n_malicious_learners=cfg.learner.n_malicious
            )
            for i in range(cfg.aggregator.n_total)
        ]

    def run_a_round(self, r):
        raise NotImplementedError()

    def run(self):
        for r in range(self.cfg.n_rounds):
            print(f"Round {r+1}/{self.cfg.n_rounds}")
            self.run_a_round(r)
            
    def exec_sql(self, sql, params):
        retries = 5
        while retries > 0:
            try:
            
                with sqlite3.connect(self.cfg.db.path) as con:
                    con.execute(sql, params)
                break
            except sqlite3.Error:
                retries -= 1
                if retries == 0:
                    raise TimeoutError("exceeds max retry times of sql query")
 