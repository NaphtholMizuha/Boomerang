from feddpr.algorithm import fetch_algorithm
from feddpr.config import toml2cfg

def main():
    alg = fetch_algorithm(toml2cfg("config/test.toml"))
    alg.run()
    
if __name__ == "__main__":
    main()