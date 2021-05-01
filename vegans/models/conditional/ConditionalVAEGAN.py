"""
ConditionalVAEGAN
-----------------
Implements the conditional variant of the Variational Autoencoder Generative Adversarial Network[1].

Trains on Kullback-Leibler loss for the latent space and attaches a adversary to get better quality output.
The Decoder acts as the generator.

Losses:
    - Encoder: Kullback-Leibler
    - Generator / Decoder: Binary cross-entropy
    - Adversary: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_KL: Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.
    - lambda_x: Weight for the reconstruction loss of the real x dimensions.

References
----------
.. [1] https://arxiv.org/pdf/1512.09300.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from vegans.models.unconditional.VAEGAN import VAEGAN
from vegans.utils.networks import Encoder, Generator, Adversary
from vegans.models.conditional.AbstractConditionalGANGAE import AbstractConditionalGANGAE

class ConditionalVAEGAN(AbstractConditionalGANGAE, VAEGAN):
    """
    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
    encoder: nn.Module
        Encoder architecture. Produces predictions in the latent space.
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
    lambda_x: float
        Weight for the reconstruction loss of the real x dimensions.
    adv_type: "Discriminator", "Critic" or "Autoencoder"
        Indicating which adversarial architecture will be used.
    feature_layer : torch.nn.*
        Output layer used to compute the feature loss. Should be from either the discriminator or critic.
        If `feature_layer` is not None, the original generator loss is replaced by a feature loss, introduced
        [here](https://arxiv.org/abs/1606.03498v1).
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
            generator,
            adversary,
            encoder,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            lambda_x=10,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./veganModels/cVAEGAN",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary, encoder=encoder,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs, adv_type=adv_type,
            feature_layer=feature_layer, fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
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
        self.lambda_x = lambda_x
        self.hyperparameters["lambda_KL"] = lambda_KL
        self.hyperparameters["lambda_x"] = lambda_x
        self.hyperparameters["adv_type"] = adv_type

        if self.secure:
            if self.encoder.output_size == self.z_dim:
                raise ValueError(
                    "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
                    "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
                )


    #########################################################################
    # Actions during training
    #########################################################################
    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
        fake_images_x = self.generate(z=Z_batch_encoded.detach(), y=y_batch)
        fake_concat_x = self.concatenate(fake_images_x, y_batch)
        fake_images_z = self.generate(z=Z_batch, y=y_batch)
        fake_concat_z = self.concatenate(fake_images_z, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return VAEGAN._calculate_generator_loss(
            self, X_batch=real_concat, Z_batch=None, fake_images_x=fake_concat_x, fake_images_z=fake_concat_z
        )

    def _calculate_encoder_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
        fake_images_x = self.generate(z=Z_batch_encoded, y=y_batch)
        fake_concat_x = self.concatenate(fake_images_x, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return VAEGAN._calculate_encoder_loss(self, X_batch=real_concat, Z_batch=None, fake_images_x=fake_concat_x)

    def _calculate_adversary_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
        fake_images_x = self.generate(z=Z_batch_encoded, y=y_batch)
        fake_concat_x = self.concatenate(fake_images_x, y_batch).detach()
        fake_images_z = self.generate(z=Z_batch, y=y_batch)
        fake_concat_z = self.concatenate(fake_images_z, y_batch).detach()
        real_concat = self.concatenate(X_batch, y_batch)
        return VAEGAN._calculate_adversary_loss(self, X_batch=real_concat, Z_batch=None, fake_images_x=fake_concat_x, fake_images_z=fake_concat_z)
