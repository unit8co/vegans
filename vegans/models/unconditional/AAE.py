"""
AAE
---
Implements the Adversarial Autoencoder[1].

Instead of using the Kullback Leibler divergence to improve the latent space distribution
we use a discriminator to determine the "realness" of the latent vector.

Losses:
    - Encoder: Binary cross-entropy + Mean-squared error
    - Generator: Mean-squared error
    - Adversary: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_z: Weight for the discriminator loss computing the realness of the latent z dimension.

References
----------
.. [1] https://arxiv.org/pdf/1511.05644.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from torch.nn import MSELoss, BCELoss, L1Loss
from vegans.utils import WassersteinLoss
from vegans.utils.networks import Encoder, Generator, Autoencoder, Adversary
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel

class AAE(AbstractGenerativeModel):
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
    lambda_z: float
        Weight for the discriminator loss computing the realness of the latent z dimension.
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
            lambda_z=10,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./veganModels/AAE",
            secure=True):

        self.adv_type = adv_type
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu, secure=secure)
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversary = Adversary(adversary, input_size=z_dim, device=device, ngpu=ngpu, adv_type=adv_type, secure=secure)
        self.neural_nets = {
            "Generator": self.generator, "Encoder": self.encoder, "Adversary": self.adversary
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
        )

        self.lambda_z = lambda_z
        self.hyperparameters["lambda_z"] = lambda_z
        self.hyperparameters["adv_type"] = adv_type

        if self.secure:
            assert self.encoder.output_size == self.z_dim, (
                "Encoder output shape must be equal to z_dim. {} vs. {}.".format(self.encoder.output_size, self.z_dim)
            )
            assert self.generator.output_size == self.x_dim, (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            loss_functions = {"Generator": MSELoss(), "Adversary": BCELoss()}
        elif self.adv_type == "Critic":
            loss_functions = {"Generator": MSELoss(), "Adversary": WassersteinLoss()}
        else:
            raise NotImplementedError("'adv_type' must be one of Discriminator or Critic.")
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x):
        return self.encoder(x)

    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Generator":
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Encoder":
            losses = self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Adversary":
            losses = self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
            losses.update(self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch))
            losses.update(self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch))
        return losses

    def _calculate_generator_loss(self, X_batch, Z_batch, fake_images=None):
        if fake_images is None:
            encoded_output = self.encode(x=X_batch).detach()
            fake_images = self.generate(encoded_output)
        gen_loss = self.loss_functions["Generator"](
            fake_images, X_batch
        )

        return {
            "Generator": gen_loss,
        }

    def _calculate_encoder_loss(self, X_batch, Z_batch, fake_images=None, encoded_output=None):
        if fake_images is None:
            encoded_output = self.encode(x=X_batch)
            fake_images = self.generate(z=encoded_output)

        if self.feature_layer is None:
            fake_predictions = self.predict(x=encoded_output)
            enc_loss_fake = self.loss_functions["Generator"](
                fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
            )
        else:
            enc_loss_fake = self._calculate_feature_loss(X_real=Z_batch, X_fake=encoded_output)
        enc_loss_reconstruction = self.loss_functions["Generator"](
            fake_images, X_batch
        )

        enc_loss = self.lambda_z*enc_loss_fake + enc_loss_reconstruction
        return {
            "Encoder": enc_loss,
            "Encoder_x": self.lambda_z*enc_loss_fake,
            "Encoder_fake": enc_loss_reconstruction,
        }

    def _calculate_adversary_loss(self, X_batch, Z_batch, encoded_output=None):
        if encoded_output is None:
            encoded_output = self.encode(x=X_batch).detach()

        fake_predictions = self.predict(x=encoded_output)
        real_predictions = self.predict(x=Z_batch)

        adv_loss_fake = self.loss_functions["Adversary"](
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )

        adv_loss = 1/2*(adv_loss_real + adv_loss_fake)
        return {
            "Adversary": adv_loss,
            "Adversary_fake": adv_loss_fake,
            "Adversary_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        }

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversary":
                if self.adv_type == "Critic":
                    for p in self.adversary.parameters():
                        p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]
