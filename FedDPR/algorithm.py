from FedDPR.config import Config
from FedDPR.training import (
    Trainer,
    fetch_dataset,
    fetch_datasplitter,
    fetch_model,
)
from FedDPR.peer import Aggregator, Attacker
from dataclasses import asdict
import json
import psycopg2
from psycopg2.extras import execute_values
import torch
from copy import deepcopy
from statistics import mean

class Algorithm:
    def __init__(self, cfg: Config):
        # Initialize the configuration object
        self.cfg = cfg

        # Insert or replace the configuration settings into the database

        params = asdict(cfg)
        params.pop("db")

        if cfg.db.reset:
            self.query_sql(
                "DELETE FROM config WHERE full_config=%s", [json.dumps(params)]
            )
            print("ID Reset")

        if cfg.db.enable:

            self.id = self.query_sql(
                """
                INSERT INTO config (model, data_het, n_c, n_s, mal_rate_c, mal_rate_s, atk_c, def_c, atk_s, def_s, full_config)
                VALUES (%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
                """,
                [
                    cfg.local.model,
                    None if cfg.split.method == 'iid' else cfg.split.alpha,
                    cfg.client.n,
                    cfg.server.n,
                    cfg.client.m / cfg.client.n,
                    cfg.server.m / cfg.server.n,
                    cfg.client.attack,
                    cfg.client.defense,
                    cfg.server.attack,
                    cfg.server.defense,
                    json.dumps(params)
                ],
            )
            print(f"ID Generated: {self.id}")

        # Fetch the training and testing datasets
        train_set, self.test_set = fetch_dataset(cfg.local.datapath, cfg.local.dataset)

        self.train_subsets = fetch_datasplitter(
            train_set, cfg.split.method, cfg.client.n, alpha=cfg.split.alpha
        ).split()

    def run(self):
        # Run the algorithm for the specified number of turns
        for t in range(self.cfg.n_turns):
            # Run the specified number of rounds for each turn
            self.reset()
            for r in range(self.cfg.n_rounds):
                print(f"Round {r + 1}/{self.cfg.n_rounds}")
                self.run_a_round(t, r)

    def query_sql(self, sql, params, is_many=False):
        # Execute SQL queries if the database is enabled
        if self.cfg.db.enable:
            with psycopg2.connect(
                f"dbname={self.cfg.db.name} user={self.cfg.db.user} password={self.cfg.db.password} host=localhost port=5432"
            ) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    if not is_many:
                        cur.execute(sql, params)
                    else:
                        execute_values(cur, sql, params)
                    if "RETURNING" in sql.upper():
                        return cur.fetchone()[0]
                    

    def reset(self):
        cfg = self.cfg
        init_model = fetch_model(self.cfg.local.model).to(self.cfg.local.device)
        models = [
            fetch_model(cfg.local.model).to(cfg.local.device)
            for _ in range(cfg.client.n)
        ]
        # Create a list of trainers, one for each client
        self.trainers = [
            Trainer(
                model=models[i],
                init_state=deepcopy(init_model.state_dict()),
                train_set=self.train_subsets[i],
                test_set=self.test_set,
                device=cfg.local.device,
                bs=cfg.local.batch_size,
                nw=cfg.local.num_workers,
                lr=cfg.local.lr,
            )
            for i in range(cfg.client.n)
        ]
        
        self.attacker = Attacker(
            n=cfg.client.n,
            m=cfg.client.m,
            method=cfg.client.attack,  
        )
        
        self.agg_servers = [
            Aggregator(
                n=cfg.client.n,
                m=cfg.client.m,
                method=cfg.server.attack if i < cfg.server.m else cfg.server.defense,
                penalty=cfg.penalty,
                device=cfg.local.device,
                is_server=True
            )
            for i in range(cfg.server.n)
        ]
        
        self.agg_clients = [
            Aggregator(
                n=cfg.server.n,
                m=cfg.server.m,
                method=cfg.client.defense,
                device=cfg.local.device,
                is_server=False
            )
            for i in range(cfg.client.n)
        ]

    def run_a_round(self, t, r):
        is_bds = (self.cfg.client.defense == 'bds') and (self.cfg.server.defense == 'bds')
        
        # step 1: locally train
        print("Client ", end="")
        for i, trainer in enumerate(self.trainers):
            trainer.local_train(self.cfg.local.n_epochs)
            print(f"{i}...", end="")
        print("Finish")
        
        
        # step 2: gather all local grads and attack
        grads_l = torch.stack([trainer.get_grad() for trainer in self.trainers])
        grads_l = self.attacker.attack(grads_l)
        
        # step 2.5: note local gradients for FedBDS
        if is_bds:
            for client, grad in zip(self.agg_clients, grads_l.unbind(0)):
                client.set_local_grad(grad)
        
        # step 3: aggregate by servers
        grads_g = torch.stack([
            server.aggregate(grads_l) for server in self.agg_servers
        ])
        
        # step 4: aggregate by clients
        for client, trainer in zip(self.agg_clients, self.trainers):
            grad = client.aggregate(grads_g)
            trainer.set_grad(grad)
            
        # step 4.5: update scores for FedBDS
        if is_bds:
            bwd_scores = torch.stack([
                client.get_bwd_scores() for client in self.agg_clients
            ]).T
            
            # print(bwd_scores)
            
            for i, (server, scores) in enumerate(zip(self.agg_servers, bwd_scores.unbind(0))):
                if i >= self.cfg.server.m:
                    server.update_fwd_scores(scores)
                    # print(server.fwd_scores)
                    
            
        
        loss, acc = zip(*[trainer.test() for trainer in self.trainers[self.cfg.client.m:]])
        loss, acc = mean(loss), mean(acc)
        print(f"Benign clients avg. loss: {loss}, acc: {acc}")
        
        if self.cfg.db.enable:
            self.query_sql(
                "INSERT INTO result VALUES (%s,%s,%s,%s,%s)",
                [self.id, t, r, loss, acc]
            )
            
            if is_bds:
                mal = self.cfg.server.m
                
                bwd_scores = torch.stack([
                    client.get_bwd_scores() for client in self.agg_clients
                ]).cpu().numpy().tolist()
                
                fwd_scores = torch.stack([
                    server.get_fwd_scores() for server in self.agg_servers[mal:]
                ]).cpu().numpy().tolist()
                
                self.query_sql("INSERT INTO score VALUES (%s,%s,%s,%s,%s)", [self.id, t, r, 'forward', fwd_scores])
                self.query_sql("INSERT INTO score VALUES (%s,%s,%s,%s,%s)", [self.id, t, r, 'backward', bwd_scores])