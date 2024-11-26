#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:55:13 2024

@author: poulimenos
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Το x εδώ πρέπει να έχει μέγεθος (batch_size, sequence_length, input_dim)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Χρησιμοποιούμε μόνο την τελευταία χρονική στιγμή
        return out


# Ορισμός του Discriminator με GRU
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Χρησιμοποιούμε μόνο την τελευταία χρονική στιγμή
        return torch.sigmoid(out)

# Ορισμός του Embedding Network (Autoencoder)
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(EmbeddingNetwork, self).__init__()
        self.encoder = nn.GRU(input_dim, latent_dim, batch_first=True)
        self.decoder = nn.GRU(latent_dim, input_dim, batch_first=True)
    
    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

# Ορισμός του GAN
class GAN:
    def __init__(self, noise_dim, data_dim, hidden_dim=128, learning_rate=0.0002):
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.generator = Generator(noise_dim, hidden_dim, data_dim)
        self.discriminator = Discriminator(data_dim, hidden_dim)
        self.embedding_network = EmbeddingNetwork(data_dim, hidden_dim)
        
        # Βελτιστοποιητές
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_E = optim.Adam(self.embedding_network.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        # Κριτήριο
        self.criterion = nn.BCELoss()

    def train(self, real_data, epochs=1000, batch_size=64):
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        for epoch in range(epochs):
            real_data_batch = torch.tensor(real_data[np.random.randint(0, real_data.shape[0], batch_size)], dtype=torch.float32)

            # Εκπαίδευση Discriminator
            # Δημιουργία θορύβου για τον Generator
            noise = torch.randn(batch_size, 1, self.noise_dim)  # 1 είναι η διάρκεια της ακολουθίας
            fake_data_batch = self.generator(noise)


            # Υπολογισμός απώλειας με πραγματικά δεδομένα
            self.optimizer_D.zero_grad()
            real_output = self.discriminator(real_data_batch)
            loss_real = self.criterion(real_output, real_labels)

            # Υπολογισμός απώλειας με ψεύτικα δεδομένα
            fake_output = self.discriminator(fake_data_batch.detach())
            loss_fake = self.criterion(fake_output, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            self.optimizer_D.step()

            # Εκπαίδευση Generator
            self.optimizer_G.zero_grad()
            fake_output = self.discriminator(fake_data_batch)
            loss_G = self.criterion(fake_output, real_labels)  # Θέλουμε ο Generator να εξαπατήσει τον Discriminator
            loss_G.backward()
            self.optimizer_G.step()

            # Εκπαίδευση Embedding Network (Autoencoder)
            self.optimizer_E.zero_grad()
            decoded = self.embedding_network(real_data_batch)
            loss_rec = nn.MSELoss()(decoded, real_data_batch)  # Απώλεια ανακατασκευής
            loss_rec.backward()
            self.optimizer_E.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}, Rec Loss: {loss_rec.item()}")

    def generate_synthetic_data(self, num_samples):
        noise = torch.randn(num_samples, self.noise_dim, self.data_dim)
        synthetic_data = self.generator(noise).detach().numpy()
        return synthetic_data




