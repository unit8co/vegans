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
Custom parameter:
    - lambda_KL: Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.

References
----------
.. [1] https://arxiv.org/pdf/1906.02691.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from vegans.utils.utils import get_input_dim
from vegans.models.unconditional.VanillaVAE import VanillaVAE
from vegans.utils.networks import Encoder, Decoder, Autoencoder
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalVanillaVAE(AbstractConditionalGenerativeModel, VanillaVAE):
    """
    Parameters
    ----------
    encoder: nn.Module
        Encoder architecture. Produces predictions in the latent space.
    decoder: nn.Module
        Decoder architecture. Produces output in the real space.
    x_dim : list, tuple
        Number of the output dimensions of the generator and input dimension of the discriminator / critic.
        In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
    z_dim : int, list, tuple
        Number of the latent dimensions for the generator input. Might have dimensions of an image.
    y_dim : int, list, tuple
        Number of dimensions for the target label. Might have dimensions of image for image to image translation, i.e.
        [nr_channels, nr_height_pixels, nr_width_pixels] or an integer representing a number of classes.
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
    lambda_KL: float
        Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.
    fixed_noise_size : int
        Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
        subdirectory, the tensorboard images tab and the samples in get_training_results().
    device : string
        Device used while training the model. Either "cpu" or "cuda".
    ngpu : int
        Number of gpus used during training if device == "cuda".
    folder : string
        Creates a folder in the current working directory with this name. All relevant files like summary, images, models and
        tensorboard output are written there. Existing folders are never overwritten or deleted. If a folder with the same name
        already exists a time stamp is appended to make it unique.
    """

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
            ngpu=0,
            folder="./CVanillaVAE",
            secure=True):

        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        dec_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        if secure:
            AbstractConditionalGenerativeModel._check_conditional_network_input(encoder, in_dim=x_dim, y_dim=y_dim, name="Encoder")
            AbstractConditionalGenerativeModel._check_conditional_network_input(decoder, in_dim=z_dim, y_dim=y_dim, name="Decoder")
        self.decoder = Decoder(decoder, input_size=dec_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.encoder = Encoder(encoder, input_size=enc_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.autoencoder = Autoencoder(self.encoder, self.decoder)
        self.neural_nets = {
            "Autoencoder": self.autoencoder
        }


        super().__init__(
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=None,
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

    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x, y=None):
        if y is None:
            x_dim = tuple(x.shape[1:])
            assert x_dim == self.adv_in_dim, (
                "If `y` is None, x must have correct shape. Given: {}. Expected: {}.".format(x_dim, self.adv_in_dim)
            )
            return VanillaVAE.encode(self, x=x)

        inpt = self.concatenate(x, y).float()
        return VanillaVAE.encode(self, x=inpt)

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        losses = self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        return losses

    def _calculate_autoencoder_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
        fake_images = self.generate(z=Z_batch_encoded, y=y_batch)
        fake_concat = self.concatenate(fake_images, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return VanillaVAE._calculate_autoencoder_loss(self, X_batch=real_concat, Z_batch=None, fake_images=fake_concat)
