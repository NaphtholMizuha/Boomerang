from FedDPR.algorithm import Algorithm
from FedDPR.config import toml2cfg
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default="config/default.toml")
    parser.add_argument('-a', "--algorithm", type=str, default=None)
    parser.add_argument('-t', "--table", type=str, default=None)
    parser.add_argument("--m-lrn", type=int, default=None)
    parser.add_argument("--m-agg", type=int, default=None)
    
    # Local training parameters
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
    parser.add_argument("--device", type=str, default=None, help="device to use (cpu/cuda)")
    
    # Data split parameters
    parser.add_argument("--split-method", type=str, default=None, 
                       choices=["iid", "dirichlet"], help="data split method")
    parser.add_argument("--split-alpha", type=float, default=None, 
                       help="alpha parameter for dirichlet split")
    
    # Database parameters
    parser.add_argument("--enable-db", action="store_true", 
                       help="enable database logging")
    parser.add_argument("--reset-db", action="store_true", 
                       help="reset database before running")
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
    
    alg = Algorithm(cfg)
    alg.run()
    
if __name__ == "__main__":
    main()
