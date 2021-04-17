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

from vegans.utils.utils import wasserstein_loss
from vegans.utils.networks import Generator, Adversariat, Encoder
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel

class BicycleGAN(AbstractGenerativeModel):
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
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            lambda_x=10,
            lambda_z=10,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder="./BicycleGAN",
            ngpu=0):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adv_type = adv_type
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu)
        self.adversariat = Adversariat(adversariat, input_size=x_dim, adv_type=adv_type, device=device, ngpu=ngpu)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu)
        self.neural_nets = {
            "Generator": self.generator, "Adversariat": self.adversariat, "Encoder": self.encoder
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
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

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            self.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss(), "L1": L1Loss(), "Reconstruction": L1Loss()}
        elif self.adv_type == "Critic":
            self.loss_functions = {"Generator": wasserstein_loss, "Adversariat": wasserstein_loss, "L1": L1Loss(), "Reconstruction": L1Loss()}
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
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Encoder":
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)

    def _calculate_generator_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(x=X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(z=Z_batch_encoded.detach())
        fake_images_z = self.generate(z=Z_batch)
        encoded_output_fake = self.encode(x=fake_images_x)
        fake_Z = self.mu(encoded_output_fake)

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

    def _calculate_encoder_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(z=Z_batch_encoded)
        fake_predictions_x = self.predict(x=fake_images_x)
        encoded_output_fake = self.encode(x=fake_images_x)
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

    def _calculate_adversariat_loss(self, X_batch, Z_batch):
        encoded_output = self.encode(x=X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch

        fake_images_x = self.generate(z=Z_batch_encoded).detach()
        fake_images_z = self.generate(z=Z_batch).detach()

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