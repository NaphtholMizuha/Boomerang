import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Trainer:
    def __init__(self, model: nn.Module, train_set: Dataset, test_set: Dataset, bs: int, nw: int, lr: float, device: str) -> None:
        self.model = model
        self.train_loader = DataLoader(train_set, batch_size=bs, num_workers=nw, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=bs, num_workers=nw)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.device = device

    def train(self):
        self.model.train()
        loss_sum = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_sum / len(self.train_loader)
    
    def train_epochs(self, n_epoch):
        for _ in range(n_epoch):
            self.train()
            
    def get_state(self):
        return self.model.state_dict()
    
    def set_state(self, state):
        self.model.load_state_dict(state)
            
    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        loss, acc = 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= len(self.test_loader)
        acc /= len(self.test_loader.dataset)
        return loss, acc