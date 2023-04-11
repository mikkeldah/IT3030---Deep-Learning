import torch 
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self, tw):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=tw, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, 2),
        )
        
        self.fcn = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fcn(x)

        return x
