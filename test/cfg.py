from FedDPR.config import toml2cfg

if __name__ == '__main__':
    cfg = toml2cfg('config/default.toml')
    print(cfg)