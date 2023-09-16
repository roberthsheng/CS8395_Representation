import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + label_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        hidden = torch.relu(self.fc1(x))
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + label_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, c):
        z = torch.cat([z, c], dim=1)
        hidden = torch.relu(self.fc1(z))
        output = torch.sigmoid(self.fc2(hidden))
        return output

# Hyperparameters
batch_size = 128
input_dim = 28*28
label_dim = 10  # Number of classes in MNIST
hidden_dim = 256
latent_dim = 2
learning_rate = 1e-3
epochs = 10

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, Optimizer
encoder = Encoder(input_dim, label_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, label_dim, hidden_dim, input_dim)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch, labels in train_loader:
        x = batch.view(-1, input_dim)
        c = torch.nn.functional.one_hot(labels, num_classes=label_dim).float()

        # Forward pass: Encoder
        mu, log_var = encoder(x, c)

        # Reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Forward pass: Decoder
        x_recon = decoder(z, c)

        # ELBO Loss
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
