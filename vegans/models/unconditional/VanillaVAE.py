"""
VAE
---
Implements the Variational Autoencoder[1].

Trains on Kullback-Leibler loss and mean squared error reconstruction loss.

Losses:
    - Encoder: Kullback-Leibler
    - Decoder: L2 (Mean Squared Error)
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://arxiv.org/pdf/1906.02691.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from torch.nn import MSELoss
from vegans.utils.networks import Encoder, Decoder, Autoencoder
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel

class VanillaVAE(AbstractGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            encoder,
            decoder,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            fixed_noise_size=32,
            device=None,
            folder="./LRGAN1v1",
            ngpu=0):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adversariat = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu)
        self.generator = Decoder(decoder, input_size=z_dim, device=device, ngpu=ngpu)
        self.autoencoder = Autoencoder(self.adversariat, self.generator)
        self.neural_nets = {
            "Autoencoder": self.autoencoder
        }

        AbstractGenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.adversariat.output_size), np.prod(z_dim))
        ).to(self.device)
        self.log_variance = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.adversariat.output_size), np.prod(z_dim))
        ).to(self.device)

        self.lambda_KL = lambda_KL
        self.hyperparameters["lambda_KL"] = lambda_KL
        assert (self.adversariat.output_size == self.z_dim), (
            "Encoder output shape must be equal to z_dim. {} vs. {}.".format(self.adversariat.output_size, self.z_dim)
        )
        assert (self.generator.output_size == self.x_dim), (
            "Decoder output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Autoencoder": MSELoss()}


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x):
        return self.adversariat(x)

    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Autoencoder":
            self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch)

    def _calculate_autoencoder_loss(self, X_batch, Z_batch):
        encoded_output = self.adversariat(X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch = mu + torch.exp(log_variance)*self.sample(n=len(X_batch))
        fake_images = self.generator(Z_batch)

        kl_loss = (log_variance**2 + mu**2 - log_variance - 1/2).sum()
        reconstruction_loss = self.loss_functions["Autoencoder"](
            fake_images, X_batch
        )

        total_loss = reconstruction_loss + self.lambda_KL*kl_loss
        self._losses.update({
            "Autoencoder": total_loss,
            "Kullback-Leibler": self.lambda_KL*kl_loss,
            "Reconstruction": reconstruction_loss,
        })