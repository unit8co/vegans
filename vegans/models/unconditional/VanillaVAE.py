"""
VanillaVAE
----------
Implements the Variational Autoencoder[1].

Trains on Kullback-Leibler loss and mean squared error reconstruction loss.

Losses:
    - Encoder: Kullback-Leibler
    - Decoder: L2 (Mean Squared Error)
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_KL: Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.

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
            folder="./VanillaVAE",
            ngpu=0,
            secure=True):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.decoder = Decoder(decoder, input_size=z_dim, device=device, ngpu=ngpu, secure=secure)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu, secure=secure)
        self.autoencoder = Autoencoder(self.encoder, self.decoder)
        self.neural_nets = {
            "Autoencoder": self.autoencoder
        }


        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
        )
        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.encoder.output_size), np.prod(z_dim))
        ).to(self.device)
        self.log_variance = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.encoder.output_size), np.prod(z_dim))
        ).to(self.device)

        self.lambda_KL = lambda_KL
        self.hyperparameters["lambda_KL"] = lambda_KL

        if self.secure:
            if self.encoder.output_size == self.z_dim:
                raise ValueError(
                    "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
                    "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
                )
            assert (self.decoder.output_size == self.x_dim), (
                "Decoder output shape must be equal to x_dim. {} vs. {}.".format(self.decoder.output_size, self.x_dim)
            )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Autoencoder": MSELoss()}


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x):
        return self.encoder(x)

    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Autoencoder":
            self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch)

    def _calculate_autoencoder_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
        fake_images = self.generate(Z_batch_encoded)

        kl_loss = 0.5*(log_variance.exp() + mu**2 - log_variance - 1).sum()
        reconstruction_loss = self.loss_functions["Autoencoder"](
            fake_images, X_batch
        )

        total_loss = reconstruction_loss + self.lambda_KL*kl_loss
        self._losses.update({
            "Autoencoder": total_loss,
            "Kullback-Leibler": self.lambda_KL*kl_loss,
            "Reconstruction": reconstruction_loss,
        })


    #########################################################################
    # Utility functions
    #########################################################################
    def sample(self, n):
        return torch.randn(n, *self.z_dim, requires_grad=True, device=self.device)