import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

def fetch_trainer(model: nn.Module, train_set: Dataset, test_set: Dataset, bs: int, nw: int, lr: float, device: str, **kwargs):
    return Trainer(model, train_set, test_set, bs, nw, lr, device, **kwargs)

class Trainer:
    def __init__(self, model: nn.Module, train_set: Dataset, test_set: Dataset, bs: int, nw: int, lr: float, device: str, **kwargs) -> None:
        self.model = model
        self.train_loader = DataLoader(train_set, batch_size=bs, num_workers=nw, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=bs, num_workers=nw)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.device = device
        self.offset = 0
        self.lr = lr
        self.backdoor = kwargs.get('backdoor')
        self.backdoor_set = kwargs.get('backdoor_set')

    def train(self):
        self.model.train()
        loss_sum = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y = (y + self.offset) % 10
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_sum / len(self.train_loader)
    
    def train_backdoor(self):
        
        self.model.train()
        loss_sum = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y = (y + self.offset) % 10
            batch_size = x.size(0)
            replace_size = batch_size // 16
            
            if replace_size > 0:
                idcs = torch.randint(0, len(self.backdoor_set), (replace_size,)).sort()[0]
                backdoor_subsets = Subset(self.backdoor_set, idcs.tolist())
                backdoor_x = torch.stack([dp[0] for dp in backdoor_subsets]).to(self.device)
                backdoor_y = torch.tensor([dp[1] for dp in backdoor_subsets]).to(self.device)

                x[-replace_size:] = backdoor_x
                y[-replace_size:] = backdoor_y
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
            
    def train_backdoor_epochs(self, n_epoch):
        for _ in range(n_epoch):
            self.train_backdoor()
            
    def get_state(self):
        return self.model.state_dict()
    
    def set_state(self, state):
        self.model.load_state_dict(state)
        
    def label_flip(self, offset):
        self.offset = offset
            
    def test(self, dataloader=None):
        self.model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        loss, acc = 0, 0

        with torch.no_grad():
            dataloader = dataloader or self.test_loader
                
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= len(self.test_loader)
        acc /= len(self.test_loader.dataset)
        return loss, acc
    
