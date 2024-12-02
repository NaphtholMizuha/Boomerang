from FedDPR.algorithm import fetch_algorithm
from FedDPR.config import toml2cfg

def main():
    alg = fetch_algorithm(toml2cfg("config/default.toml"))
    alg.run()
    
if __name__ == "__main__":
    main()