from feddpr.training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from feddpr.peer import Aggregator, Learner
from copy import deepcopy
from tqdm import tqdm
import duckdb
import datetime

if __name__ == "__main__":
    n_client = 50
    n_round = 50
    n_epoch = 5
    lr = 0.05
    data_path = "~/data"
    model_name = "cnn"
    data_name = "cifar10"
    split_name = "iid"
    device = "cuda"
    
    with open("config/create.sql", "r") as f:
        sql = f.read()
    
    with duckdb.connect("result.db") as con:
        con.execute(sql)

    models = [fetch_model(model_name).to(device) for _ in range(n_client)]
    print(f"Model: {model_name}")
    train_set, test_set = fetch_dataset(data_path, data_name)
    print(f"Dataset: {data_name}")
    splitter = fetch_datasplitter(train_set, split_name, n_client)
    print(f"Data Splitter: {split_name} with {n_client} clients")
    train_subsets = splitter.split()
    print(f"Train Subsets: {len(train_subsets)}")
    trainers = [
        Trainer(
            model=models[i],
            train_set=train_subsets[i],
            test_set=test_set,
            device="cuda",
            bs=32,
            nw=2,
            lr=lr,
        )
        for i in range(n_client)
    ]
    workers = [
        Learner(
            n_epoch=n_epoch,
            init_state=deepcopy(models[i].state_dict()),
            trainer=trainers[i],
        )
        for i in range(n_client)
    ]
    aggregator = Aggregator(n_client)

    for r in range(n_round):
        print(f"Round {r+1}/{n_round}")
        grads = []
        for worker in tqdm(workers):
            worker.local_train()
            grads.append(worker.get_weight())

        grad_g = aggregator.aggregate(grads)

        for worker in workers:
            i = workers.index(worker)
            worker.set_weight(grad_g)

        loss, acc = workers[0].test()
        print(f"Loss: {loss}, Acc: {acc}")

        with duckdb.connect("result.db") as con:
            con.execute(
                "INSERT INTO fedavg VALUES (?, ?, ?, ?)",
                [datetime.datetime.now(), r, loss, acc],
            )
