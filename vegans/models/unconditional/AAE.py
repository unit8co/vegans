"""
AAE
---
Implements the Adversarial Autoencoder[1].

Instead of using the Kullback Leibler divergence to improve the latent space distribution
we use a discriminator to determine the "realness" of the latent vector.

Losses:
    - Encoder: Kullback-Leibler
    - Decoder: Binary cross-entropy
    - Adversariat: Binary cross-entropy
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
from vegans.utils.utils import WassersteinLoss
from vegans.utils.networks import Encoder, Generator, Autoencoder, Adversariat
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel

class AAE(AbstractGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            encoder,
            generator,
            adversariat,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            lambda_z=10,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder="./AAE",
            ngpu=0,
            secure=True):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adv_type = adv_type
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu, secure=secure)
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu, secure=secure)
        self.autoencoder = Autoencoder(self.encoder, self.generator)
        self.adversariat = Adversariat(adversariat, input_size=z_dim, device=device, ngpu=ngpu, adv_type=adv_type, secure=secure)
        self.neural_nets = {
            "Generator": self.generator, "Encoder": self.encoder, "Adversariat": self.adversariat
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
        )

        self.lambda_z = lambda_z
        self.hyperparameters["lambda_z"] = lambda_z
        self.hyperparameters["adv_type"] = adv_type

        if self.secure:
            assert (self.encoder.output_size == self.z_dim), (
                "Encoder output shape must be equal to z_dim. {} vs. {}.".format(self.encoder.output_size, self.z_dim)
            )
            assert (self.generator.output_size == self.x_dim), (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            self.loss_functions = {"Generator": MSELoss(), "Adversariat": BCELoss()}
        elif self.adv_type == "Critic":
            self.loss_functions = {"Generator": MSELoss(), "Adversariat": WassersteinLoss()}
        else:
            raise NotImplementedError("'adv_type' must be one of Discriminator or Critic.")


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x):
        return self.encoder(x)

    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Encoder":
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)

    def _calculate_generator_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(x=X_batch).detach()
        fake_images = self.generate(encoded_output)
        gen_loss = self.loss_functions["Generator"](
            fake_images, X_batch
        )

        self._losses.update({
            "Generator": gen_loss,
        })

    def _calculate_encoder_loss(self, X_batch, Z_batch):
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
        self._losses.update({
            "Encoder": enc_loss,
            "Encoder_x": self.lambda_z*enc_loss_fake,
            "Encoder_fake": enc_loss_reconstruction,
        })

    def _calculate_adversariat_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(x=X_batch).detach()

        fake_predictions = self.predict(x=encoded_output)
        real_predictions = self.predict(x=Z_batch)

        adv_loss_fake = self.loss_functions["Adversariat"](
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversariat"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )

        adv_loss = 1/3*(adv_loss_real + adv_loss_fake)
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        })

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversariat":
                if self.adv_type == "Critic":
                    for p in self.adversariat.parameters():
                        p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]
