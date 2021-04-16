"""
VAEGAN
------
Implements the Variational Autoencoder Generative Adversarial Network[1].

Trains on Kullback-Leibler loss for the latent space and attaches a adversariat to get better quality output.
The Decoder acts as the generator.

Losses:
    - Encoder: Kullback-Leibler
    - Decoder: Binary cross-entropy
    - Adversariat: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://arxiv.org/pdf/1512.09300.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from torch.nn import MSELoss, BCELoss
from vegans.utils.networks import Encoder, Generator, Autoencoder, Adversariat
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel
from vegans.utils.utils import wasserstein_loss

class VAEGAN(AbstractGenerativeModel):
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
            lambda_KL=10,
            adv_type="Discriminator",
            fixed_noise_size=32,
            device=None,
            folder="./LRGAN1v1",
            ngpu=0):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adv_type = adv_type
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu)
        self.autoencoder = Autoencoder(self.encoder, self.generator)
        self.adversariat = Adversariat(adversariat, input_size=x_dim, device=device, ngpu=ngpu, adv_type=adv_type)
        self.neural_nets = {
            "Generator": self.generator, "Encoder": self.encoder, "Adversariat": self.adversariat
        }

        super().__init__(
            self, x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs,
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
        self.hyperparameters["adv_type"] = adv_type
        if self.encoder.output_size == self.z_dim:
            raise ValueError(
                "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
                "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            )
        assert (self.generator.output_size == self.x_dim), (
            "Decoder output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            self.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss()}
        elif self.adv_type == "Critic":
            self.loss_functions = {"Generator": wasserstein_loss, "Adversariat": wasserstein_loss}
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
        encoded_output = self.encode(X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(Z_batch_encoded.detach())
        fake_images_z = self.generate(Z_batch)

        fake_predictions_x = self.predict(x=fake_images_x)
        fake_predictions_z = self.predict(x=fake_images_z)

        gen_loss_fake_x = self.loss_functions["Generator"](
            fake_predictions_x, torch.ones_like(fake_predictions_x, requires_grad=False)
        )
        gen_loss_fake_z = self.loss_functions["Generator"](
            fake_predictions_z, torch.ones_like(fake_predictions_z, requires_grad=False)
        )

        gen_loss = 0.5*(gen_loss_fake_x + gen_loss_fake_z)
        self._losses.update({
            "Generator": gen_loss,
            "Generator_x": gen_loss_fake_x,
            "Generator_z": gen_loss_fake_z,
        })

    def _calculate_encoder_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)

        kl_loss = 0.5*(log_variance.exp() + mu**2 - log_variance - 1).sum()
        enc_loss = self.lambda_KL*kl_loss
        self._losses.update({
            "Encoder": enc_loss,
        })

    def _calculate_adversariat_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(Z_batch_encoded).detach()
        fake_images_z = self.generate(Z_batch).detach()

        fake_predictions_x = self.predict(x=fake_images_x)
        fake_predictions_z = self.predict(x=fake_images_z)
        real_predictions = self.predict(x=X_batch)

        adv_loss_fake_x = self.loss_functions["Adversariat"](
            fake_predictions_x, torch.zeros_like(fake_predictions_x, requires_grad=False)
        )
        adv_loss_fake_z = self.loss_functions["Adversariat"](
            fake_predictions_z, torch.zeros_like(fake_predictions_z, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversariat"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )

        adv_loss = 1/3*(adv_loss_fake_z + adv_loss_fake_x + adv_loss_real)
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake_x": adv_loss_fake_x,
            "Adversariat_fake_z": adv_loss_fake_z,
            "Adversariat_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake_x
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


    #########################################################################
    # Utility functions
    #########################################################################
    def sample(self, n):
        return torch.randn(n, *self.z_dim, requires_grad=True, device=self.device)