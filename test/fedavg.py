from FedDPR.algorithm import fetch_algorithm
from FedDPR.config import toml2cfg

def main():
    cfg = toml2cfg("config/default.toml")
    cfg.algorithm = 'fedavg'
    alg = fetch_algorithm(cfg)
    alg.run()
    
if __name__ == "__main__":
    main()