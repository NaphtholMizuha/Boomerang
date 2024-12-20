import torch.nn as nn

class CnnGray(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[120, 84], out_dim=10, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6 , 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], out_dim),         
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))

class Cnn(nn.Module):
    def __init__(self, input_dim=400, hidden_dims=[120, 84], out_dim=10, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6 , 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], out_dim),         
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))