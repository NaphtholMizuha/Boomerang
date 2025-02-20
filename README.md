# FedDPR - Decentralized Private Robust Federated Learning System

## Introduction

FedDPR is a decentralized, privacy-preserving, and robust federated learning framework. It provides:

- **Decentralized Architecture**: Peer-to-peer communication without central server
- **Privacy Protection**: Differential privacy mechanisms
- **Robustness**: Byzantine fault tolerance
- **Flexibility**: Support multiple datasets and models

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- SQLite3

### Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.toml` to customize:

- Network settings
- Privacy parameters
- Training hyperparameters
- Dataset selection

## Database Setup

Initialize database:
```bash
python db/create/record.sql
python db/create/score.sql 
python db/create/setting.sql
```


## Running the System

Start training:
```bash
python run.py --config config/config.yaml
```

Batch training:
```bash
python batch.py --config config/config.yaml
```

Query results:
```bash
python query.py --config config/config.yaml
```

## Project Structure

```
FedDPR/
├── algorithm/        # Core algorithms
│   ├── base.py       # Base algorithm class
│   ├── fl.py         # Federated learning implementations
│   └── factory.py    # Algorithm factory
├── peer/             # Node implementations
│   ├── aggregator.py # Aggregator node
│   └── learner.py    # Learner node
├── training/         # Training framework
│   ├── dataset/      # Dataset handlers
│   ├── datasplitter/ # Data partitioning strategies
│   └── model/        # Model implementations
└── utils.py          # Utility functions
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.