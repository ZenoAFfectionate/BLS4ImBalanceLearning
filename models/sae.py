import torch
import torch.nn as nn


class ConvSparseAutoencoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, sparsity_target, sparsity_weight):
        """ Sparse AutoEncoder based on convolution operation """

        super(ConvSparseAutoencoder, self).__init__()
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        
        # encoder structure
        self.encoder = nn.Sequential(
            # conv layer 1
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # conv layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            # conv layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # bottleneck layer
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # decoder structure
        self.decoder = nn.Sequential(
            # transpose conv layer 1
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # transpose conv layer 2
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            # transpose conv layer 3
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # output layer
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # model weight initializetion
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        # encode data into feature
        encoded = self.encoder(x)
        # decode feature back to data
        decoded = self.decoder(encoded)
        return decoded, encoded


    def kl_divergence(self, activations):
        """ calculate KL-divergence sparsity penality """
        # activation size: (batch_size, hidden_dim, H, W)

        # calculate the average activation for each feature map
        rho_hat = torch.mean(activations, dim=[0, 2, 3])
        
        # avoid numeric instability (log(0))
        rho_hat = torch.clamp(rho_hat, min=1e-5, max=1-1e-5)
        rho = torch.tensor(self.sparsity_target)
        
        # calculate KL divergence: ρ*log(ρ/ρ_hat) + (1-ρ)*log((1-ρ)/(1-ρ_hat))
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return torch.sum(kl)
