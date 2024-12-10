from FedDPR.algorithm import fetch_algorithm
from FedDPR.config import toml2cfg
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default="config/default.toml")
    parser.add_argument('-a', "--algorithm", type=str, default=None)
    parser.add_argument('-t', "--table", type=str, default=None)
    parser.add_argument("--m-lrn", type=int, default=None)
    parser.add_argument("--m-agg", type=int, default=None)
    args = parser.parse_args()
    
    cfg = toml2cfg(args.config)
    
    if args.algorithm is not None:
        cfg.algorithm = args.algorithm
        
    if args.table is not None:
        cfg.db.table = args.table
        
    if args.m_lrn is not None:
        cfg.learner.n_malicious = args.m_lrn
        
    if args.m_agg is not None:
        cfg.aggregator.n_malicious = args.m_agg
    
    alg = fetch_algorithm(cfg)
    alg.run()
    
if __name__ == "__main__":
    main()