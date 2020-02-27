import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, latent_size: int, output_size: int = 3):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.PReLU(), 
            nn.Linear(128, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.PReLU(), 
            nn.Linear(128, input_size)
        )
    
    def encode(self, inputs):
        return self.encoder(inputs)
    
    def decode(self, latent_vector):
        return self.decoder(latent_vector)

    def forward(self, inputs):
        return self.decode(self.encode(inputs))
