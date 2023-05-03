import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim=16):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 7x7
            nn.ReLU(),   
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 4x4
            nn.ReLU(),   
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 2x2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*2*2, 128)   
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Unflatten(1, (64, 2, 2)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()        
        )


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        sample = mu + (eps * std) 
        return sample

    
    def forward(self, x):
        # encode
        x = self.encoder(x) # 128

        # get my and log var
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        # get latent vector through reparam
        z = self.reparameterize(mu, log_var)

        # decode
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var