"""
BicycleGAN
----------
Implements the BicycleGAN[1], a combination of the VAEGAN and the LRGAN.

It utilizes both steps of the Variational Autoencoder (Kullback-Leibler Loss) and uses the same
encoder architecture for the latent regression of generated images.

Losses:
    - Generator: Binary cross-entropy + L1-latent-loss + L1-reconstruction-loss
    - Discriminator: Binary cross-entropy
    - Encoder: Kullback-Leibler Loss + L1-latent-loss + L1-reconstruction-loss
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://arxiv.org/pdf/1711.11586.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from torch.nn import BCELoss, L1Loss
from torch.nn import MSELoss as L2Loss

from vegans.utils.utils import get_input_dim
from vegans.utils.networks import Generator, Adversariat, Encoder
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalBicycleGAN(AbstractConditionalGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            encoder,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            lambda_x=10,
            lambda_z=10,
            fixed_noise_size=32,
            device=None,
            folder="./LRGAN1v1",
            ngpu=0):

        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        AbstractConditionalGenerativeModel._check_conditional_network_input(encoder, in_dim=x_dim, y_dim=y_dim, name="Encoder")
        AbstractConditionalGenerativeModel._check_conditional_network_input(generator, in_dim=z_dim, y_dim=y_dim, name="Generator")
        AbstractConditionalGenerativeModel._check_conditional_network_input(adversariat, in_dim=x_dim, y_dim=y_dim, name="Adversariat")
        self.encoder = Encoder(encoder, input_size=enc_in_dim, device=device, ngpu=ngpu)
        self.generator = Generator(generator, input_size=gen_in_dim, device=device, ngpu=ngpu)
        self.adversariat = Adversariat(adversariat, input_size=adv_in_dim, adv_type="Discriminator", device=device, ngpu=ngpu)
        self.neural_nets = {
            "Generator": self.generator, "Adversariat": self.adversariat, "Encoder": self.encoder
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
        self.lambda_x = lambda_x
        self.lambda_z = lambda_z
        self.hyperparameters["lambda_KL"] = lambda_KL
        self.hyperparameters["lambda_x"] = lambda_x
        self.hyperparameters["lambda_z"] = lambda_z
        assert (self.generator.output_size == self.x_dim), (
            "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
        )
        if self.encoder.output_size == self.z_dim:
            raise ValueError(
                "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
                "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss(), "L1": L1Loss(), "Reconstruction": L1Loss()}


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x, y):
        inpt = self.concatenate(x, y).float()
        return self.encoder(inpt)

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Encoder":
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(z=Z_batch_encoded.detach(), y=y_batch)
        fake_images_z = self.generate(z=Z_batch, y=y_batch)

        fake_predictions_x = self.predict(x=fake_images_x, y=y_batch)
        fake_predictions_z = self.predict(x=fake_images_z, y=y_batch)
        encoded_output_fake = self.encode(x=fake_images_x, y=y_batch)
        fake_Z = self.mu(encoded_output_fake)

        gen_loss_fake_x = self.loss_functions["Generator"](
            fake_predictions_x, torch.ones_like(fake_predictions_x, requires_grad=False)
        )
        gen_loss_fake_z = self.loss_functions["Generator"](
            fake_predictions_z, torch.ones_like(fake_predictions_z, requires_grad=False)
        )
        gen_loss_reconstruction_x = self.loss_functions["Reconstruction"](
            fake_images_x, X_batch
        )
        gen_loss_reconstruction_z = self.loss_functions["Reconstruction"](
            fake_Z, Z_batch
        )

        gen_loss = 1/4*(gen_loss_fake_x + gen_loss_fake_z + self.lambda_x*gen_loss_reconstruction_x + self.lambda_z*gen_loss_reconstruction_z)
        self._losses.update({
            "Generator": gen_loss,
            "Generator_x": gen_loss_fake_x,
            "Generator_z": gen_loss_fake_z,
            "Reconstruction_x": self.lambda_x*gen_loss_reconstruction_x,
            "Reconstruction_z": self.lambda_z*gen_loss_reconstruction_z
        })

    def _calculate_encoder_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(z=Z_batch_encoded, y=y_batch)
        fake_predictions_x = self.predict(x=fake_images_x, y=y_batch)
        encoded_output_fake = self.encode(x=fake_images_x, y=y_batch)
        fake_Z = self.mu(encoded_output_fake)

        enc_loss_fake_x = self.loss_functions["Generator"](
            fake_predictions_x, torch.ones_like(fake_predictions_x, requires_grad=False)
        )
        enc_loss_reconstruction_x = self.loss_functions["Reconstruction"](
            fake_images_x, X_batch
        )
        enc_loss_reconstruction_z = self.loss_functions["Reconstruction"](
            fake_Z, Z_batch
        )
        kl_loss = 0.5*(log_variance.exp() + mu**2 - log_variance - 1).sum()

        enc_loss = enc_loss_fake_x + self.lambda_KL*kl_loss + self.lambda_x*enc_loss_reconstruction_x + self.lambda_z*enc_loss_reconstruction_z
        self._losses.update({
            "Encoder": enc_loss,
            "Encoder_x": enc_loss_fake_x,
            "Kullback-Leibler": self.lambda_KL*kl_loss,
            "Reconstruction_x": self.lambda_x*enc_loss_reconstruction_x,
            "Reconstruction_z": self.lambda_z*enc_loss_reconstruction_z
        })

    def _calculate_adversariat_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(z=Z_batch_encoded, y=y_batch).detach()
        fake_images_z = self.generate(z=Z_batch, y=y_batch).detach()

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