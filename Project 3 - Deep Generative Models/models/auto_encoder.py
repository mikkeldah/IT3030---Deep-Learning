import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, encoder_dim=16):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1), # Output size: [16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output size: [32, 7, 7]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output size: [64, 4, 4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output size: [128, 2, 2]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*2*2, 128), 
            nn.ReLU(),
            nn.Linear(128, encoder_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128*2*2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2 , 2)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x