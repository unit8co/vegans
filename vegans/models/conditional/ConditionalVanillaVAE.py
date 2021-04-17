"""
ConditionalVanillaVAE
---------------------
Implements the conditional variant of the Variational Autoencoder[1].

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
from vegans.utils.utils import get_input_dim
from vegans.utils.networks import Encoder, Decoder, Autoencoder
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalVanillaVAE(AbstractConditionalGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            encoder,
            decoder,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            fixed_noise_size=32,
            device=None,
            folder="./CVanillaVAE",
            ngpu=0):

        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        dec_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        AbstractConditionalGenerativeModel._check_conditional_network_input(encoder, in_dim=x_dim, y_dim=y_dim, name="Encoder")
        AbstractConditionalGenerativeModel._check_conditional_network_input(decoder, in_dim=z_dim, y_dim=y_dim, name="Decoder")
        self.decoder = Decoder(decoder, input_size=dec_in_dim, device=device, ngpu=ngpu)
        self.encoder = Encoder(encoder, input_size=enc_in_dim, device=device, ngpu=ngpu)
        self.autoencoder = Autoencoder(self.encoder, self.decoder)
        self.neural_nets = {
            "Autoencoder": self.autoencoder
        }


        super().__init__(
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
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
    def encode(self, x, y):
        inpt = self.concatenate(x, y).float()
        return self.encoder(inpt)

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        if who == "Autoencoder":
            self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)

    def _calculate_autoencoder_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
        fake_images = self.generate(z=Z_batch_encoded, y=y_batch)

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