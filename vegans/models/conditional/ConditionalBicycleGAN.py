"""
ConditionalBicycleGAN
---------------------
Implements the BicycleGAN[1], a combination of the VAEGAN and the LRGAN.

It utilizes both steps of the Variational Autoencoder (Kullback-Leibler Loss) and uses the same
encoder architecture for the latent regression of generated images.

Losses:
    - Generator: Binary cross-entropy + L1-latent-loss + L1-reconstruction-loss
    - Discriminator: Binary cross-entropy
    - Encoder: Kullback-Leibler Loss + L1-latent-loss + L1-reconstruction-loss
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_KL: Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.
    - lambda_x: Weight for the reconstruction loss of the real x dimensions.
    - lambda_z: Weight for the reconstruction loss of the latent z dimensions.

References
----------
.. [1] https://arxiv.org/pdf/1711.11586.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from vegans.utils.layers import LayerReshape
from vegans.utils.utils import get_input_dim
from vegans.utils.utils import WassersteinLoss
from vegans.models.unconditional.BicycleGAN import BicycleGAN
from vegans.utils.networks import Generator, Adversary, Encoder
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalBicycleGAN(AbstractConditionalGenerativeModel, BicycleGAN):
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
    lambda_z: float
        Weight for the reconstruction loss of the latent z dimensions.
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
            lambda_z=10,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./CBicycleGAN",
            secure=True):

        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        if secure:
            AbstractConditionalGenerativeModel._check_conditional_network_input(encoder, in_dim=x_dim, y_dim=y_dim, name="Encoder")
            AbstractConditionalGenerativeModel._check_conditional_network_input(generator, in_dim=z_dim, y_dim=y_dim, name="Generator")
            AbstractConditionalGenerativeModel._check_conditional_network_input(adversary, in_dim=x_dim, y_dim=y_dim, name="Adversary")
        self.adv_type = adv_type
        self.encoder = Encoder(encoder, input_size=enc_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.generator = Generator(generator, input_size=gen_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversary = Adversary(adversary, input_size=adv_in_dim, adv_type=adv_type, device=device, ngpu=ngpu, secure=secure)
        self.neural_nets = {
            "Generator": self.generator, "Adversary": self.adversary, "Encoder": self.encoder
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
        )
        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.encoder.output_size), np.prod(z_dim)),
            LayerReshape(z_dim)
        ).to(self.device)
        self.log_variance = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.encoder.output_size), np.prod(z_dim)),
            LayerReshape(z_dim)
        ).to(self.device)

        self.lambda_KL = lambda_KL
        self.lambda_x = lambda_x
        self.lambda_z = lambda_z
        self.hyperparameters["lambda_KL"] = lambda_KL
        self.hyperparameters["lambda_x"] = lambda_x
        self.hyperparameters["lambda_z"] = lambda_z

        if self.secure:
            assert (self.generator.output_size == self.x_dim), (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )
            # TODO: Remove those lines or use them again, but not commented
            # if self.encoder.output_size == self.z_dim:
            #     raise ValueError(
            #         "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
            #         "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            #     )


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x, y=None):
        if y is None:
            x_dim = tuple(x.shape[1:])
            assert x_dim == self.adv_in_dim, (
                "If `y` is None, x must have correct shape. Given: {}. Expected: {}.".format(x_dim, self.adv_in_dim)
            )
            return BicycleGAN.encode(self, x=x)

        inpt = self.concatenate(x, y).float()
        return BicycleGAN.encode(self, x=inpt)

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        if who == "Generator":
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Adversary":
            losses = self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Encoder":
            losses = self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            losses.update(self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch))
            losses.update(self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch))
        return losses

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
        return BicycleGAN._calculate_generator_loss(
            self, X_batch=real_concat, Z_batch=Z_batch, fake_images_x=fake_concat_x, fake_images_z=fake_concat_z
        )

    def _calculate_encoder_loss(self, X_batch, Z_batch, y_batch):
        encoded_output = self.encode(x=X_batch, y=y_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)
        Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
        fake_images_x = self.generate(z=Z_batch_encoded, y=y_batch)
        fake_concat_x = self.concatenate(fake_images_x, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return BicycleGAN._calculate_encoder_loss(
            self, X_batch=real_concat, Z_batch=Z_batch, fake_images_x=fake_concat_x
        )

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
        return BicycleGAN._calculate_adversary_loss(
            self, X_batch=real_concat, Z_batch=Z_batch, fake_images_x=fake_concat_x, fake_images_z=fake_concat_z
        )
