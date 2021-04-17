"""
ConditionalVAEGAN
-----------------
Implements the conditional variant of the Variational Autoencoder Generative Adversarial Network[1].

Trains on Kullback-Leibler loss for the latent space and attaches a adversariat to get better quality output.
The Decoder acts as the generator.

Losses:
    - Encoder: Kullback-Leibler
    - Generator / Decoder: Binary cross-entropy
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

from vegans.utils.utils import get_input_dim
from torch.nn import MSELoss, BCELoss, L1Loss
from vegans.utils.utils import wasserstein_loss
from vegans.utils.networks import Encoder, Generator, Autoencoder, Adversariat
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalVAEGAN(AbstractConditionalGenerativeModel):
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
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            adv_type="Discriminator",
            fixed_noise_size=32,
            device=None,
            folder="./CVAEGAN",
            ngpu=0):

        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        AbstractConditionalGenerativeModel._check_conditional_network_input(encoder, in_dim=x_dim, y_dim=y_dim, name="Encoder")
        AbstractConditionalGenerativeModel._check_conditional_network_input(generator, in_dim=z_dim, y_dim=y_dim, name="Generator")
        AbstractConditionalGenerativeModel._check_conditional_network_input(adversariat, in_dim=x_dim, y_dim=y_dim, name="Adversariat")
        self.adv_type = adv_type
        self.encoder = Encoder(encoder, input_size=enc_in_dim, device=device, ngpu=ngpu)
        self.generator = Generator(generator, input_size=gen_in_dim, device=device, ngpu=ngpu)
        self.autoencoder = Autoencoder(self.encoder, self.generator)
        self.adversariat = Adversariat(adversariat, input_size=adv_in_dim, device=device, ngpu=ngpu, adv_type=adv_type)
        self.neural_nets = {
            "Generator": self.generator, "Encoder": self.encoder, "Adversariat": self.adversariat
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
            self.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss(), "Reconstruction": L1Loss()}
        elif self.adv_type == "Critic":
            self.loss_functions = {"Generator": wasserstein_loss, "Adversariat": wasserstein_loss, "Reconstruction": L1Loss()}
        else:
            raise NotImplementedError("'adv_type' must be one of Discriminator or Critic.")


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x, y):
        inpt = self.concatenate(x, y).float()
        return self.encoder(inpt)

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Encoder":
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(y=y_batch, z=Z_batch_encoded.detach())
        fake_images_z = self.generate(y=y_batch, z=Z_batch)

        fake_predictions_x = self.predict(x=fake_images_x, y=y_batch)
        fake_predictions_z = self.predict(x=fake_images_z, y=y_batch)

        gen_loss_fake_x = self.loss_functions["Generator"](
            fake_predictions_x, torch.ones_like(fake_predictions_x, requires_grad=False)
        )
        gen_loss_fake_z = self.loss_functions["Generator"](
            fake_predictions_z, torch.ones_like(fake_predictions_z, requires_grad=False)
        )
        gen_loss_reconstruction = self.loss_functions["Reconstruction"](
            fake_images_x, X_batch
        )

        gen_loss = 1/3*(gen_loss_fake_x + gen_loss_fake_z + self.lambda_x*gen_loss_reconstruction)
        self._losses.update({
            "Generator": gen_loss,
            "Generator_x": gen_loss_fake_x,
            "Generator_z": gen_loss_fake_z,
            "Reconstruction": gen_loss_reconstruction
        })

    def _calculate_encoder_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(z=Z_batch_encoded, y=y_batch)
        fake_predictions_x = self.predict(x=fake_images_x, y=y_batch)

        enc_loss_fake_x = self.loss_functions["Generator"](
            fake_predictions_x, torch.ones_like(fake_predictions_x, requires_grad=False)
        )
        enc_loss_reconstruction = self.loss_functions["Reconstruction"](
            fake_images_x, X_batch
        )
        kl_loss = 0.5*(log_variance.exp() + mu**2 - log_variance - 1).sum()

        enc_loss = 1/3*(enc_loss_fake_x + self.lambda_KL*kl_loss + self.lambda_x*enc_loss_reconstruction)
        self._losses.update({
            "Encoder": enc_loss,
            "Encoder_x": enc_loss_fake_x,
            "Kullback-Leibler": self.lambda_KL*kl_loss,
            "Reconstruction": self.lambda_x*enc_loss_reconstruction
        })

    def _calculate_adversariat_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(y=y_batch, z=Z_batch_encoded).detach()
        fake_images_z = self.generate(y=y_batch, z=Z_batch).detach()

        fake_predictions_x = self.predict(x=fake_images_x, y=y_batch)
        fake_predictions_z = self.predict(x=fake_images_z, y=y_batch)
        real_predictions = self.predict(x=X_batch, y=y_batch)

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