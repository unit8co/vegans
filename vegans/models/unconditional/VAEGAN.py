"""
VAEGAN
------
Implements the Variational Autoencoder Generative Adversarial Network[1].

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

from torch.nn import L1Loss
from vegans.utils.networks import Encoder, Generator, Adversary
from vegans.models.unconditional.AbstractGANGAE import AbstractGANGAE

class VAEGAN(AbstractGANGAE):
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
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            lambda_x=10,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./VAEGAN",
            secure=True):


        super().__init__(
            generator=generator, adversary=adversary, encoder=encoder,
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, adv_type=adv_type, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, ngpu=ngpu, folder=folder, secure=secure
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

        if self.secure:
            assert self.encoder.output_size != self.z_dim, (
                "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
                "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            )

    def _define_loss(self):
        loss_functions = super()._define_loss()
        loss_functions.update({"Reconstruction": L1Loss()})
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def _calculate_generator_loss(self, X_batch, Z_batch, fake_images_x=None, fake_images_z=None):
        if fake_images_x is None:
            encoded_output = self.encode(x=X_batch)
            mu = self.mu(encoded_output)
            log_variance = self.log_variance(encoded_output)
            Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
            fake_images_x = self.generate(z=Z_batch_encoded.detach())
        if fake_images_z is None:
            fake_images_z = self.generate(Z_batch)

        if self.feature_layer is None:
            fake_predictions_x = self.predict(x=fake_images_x)
            fake_predictions_z = self.predict(x=fake_images_z)

            gen_loss_fake_x = self.loss_functions["Generator"](
                fake_predictions_x, torch.ones_like(fake_predictions_x, requires_grad=False)
            )
            gen_loss_fake_z = self.loss_functions["Generator"](
                fake_predictions_z, torch.ones_like(fake_predictions_z, requires_grad=False)
            )
        else:
            gen_loss_fake_x = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images_x)
            gen_loss_fake_z = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images_z)
        gen_loss_reconstruction = self.loss_functions["Reconstruction"](
            fake_images_x, X_batch
        )

        gen_loss = 1/3*(gen_loss_fake_x + gen_loss_fake_z + self.lambda_x*gen_loss_reconstruction)
        return {
            "Generator": gen_loss,
            "Generator_x": gen_loss_fake_x,
            "Generator_z": gen_loss_fake_z,
            "Reconstruction": gen_loss_reconstruction
        }

    def _calculate_encoder_loss(self, X_batch, Z_batch, fake_images_x=None):
        encoded_output = self.encode(x=X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        if fake_images_x is None:
            Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
            fake_images_x = self.generate(z=Z_batch_encoded)

        fake_predictions_x = self.predict(x=fake_images_x)

        enc_loss_fake_x = self.loss_functions["Generator"](
            fake_predictions_x, torch.ones_like(fake_predictions_x, requires_grad=False)
        )
        enc_loss_reconstruction = self.loss_functions["Reconstruction"](
            fake_images_x, X_batch
        )
        kl_loss = 0.5*(log_variance.exp() + mu**2 - log_variance - 1).sum()

        enc_loss = 1/3*(enc_loss_fake_x + self.lambda_KL*kl_loss + self.lambda_x*enc_loss_reconstruction)
        return {
            "Encoder": enc_loss,
            "Encoder_x": enc_loss_fake_x,
            "Kullback-Leibler": self.lambda_KL*kl_loss,
            "Reconstruction": self.lambda_x*enc_loss_reconstruction
        }

    def _calculate_adversary_loss(self, X_batch, Z_batch, fake_images_x=None, fake_images_z=None):
        if fake_images_x is None:
            encoded_output = self.encode(x=X_batch)
            mu = self.mu(encoded_output)
            log_variance = self.log_variance(encoded_output)
            Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
            fake_images_x = self.generate(z=Z_batch_encoded).detach()
        if fake_images_z is None:
            fake_images_z = self.generate(Z_batch).detach()

        fake_predictions_x = self.predict(x=fake_images_x)
        fake_predictions_z = self.predict(x=fake_images_z)
        real_predictions = self.predict(x=X_batch)

        adv_loss_fake_x = self.loss_functions["Adversary"](
            fake_predictions_x, torch.zeros_like(fake_predictions_x, requires_grad=False)
        )
        adv_loss_fake_z = self.loss_functions["Adversary"](
            fake_predictions_z, torch.zeros_like(fake_predictions_z, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )

        adv_loss = 1/3*(adv_loss_fake_z + adv_loss_fake_x + adv_loss_real)
        return {
            "Adversary": adv_loss,
            "Adversary_fake_x": adv_loss_fake_x,
            "Adversary_fake_z": adv_loss_fake_z,
            "Adversary_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake_x
        }
