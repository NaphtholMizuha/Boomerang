# Boomerang - Multi-server Private Robust Federated Learning System

## Introduction

Boomerang is privacy-preserving and robust federated learning framework. It provides:

- **Multi-server Architecture**: Avoid single-point failure and poisoning
- **Privacy Protection**: Support HE
- **Robustness**: Filter malicious clients and servers
- **Flexibility**: Support multiple datasets and models


## Configuration

Edit `sample_config.toml` to customize.


## Database Setup

run all `.db` files in `./db` directory.


## Running the System

Start training:
```bash
python run.py --config [your config]
```

Batch training:
```bash
python batch.py


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.