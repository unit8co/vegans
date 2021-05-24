"""
ConditionalInfoGAN
------------------
Implements the InfoGAN[1].

It introduces an encoder network which maps the generator output back to the latent
input space. This should help to prevent mode collapse and improve image variety.

Losses:
    - Generator: Binary cross-entropy + Normal Log-Likelihood + Multinomial Log-Likelihood
    - Discriminator: Binary cross-entropy
    - Encoder: Normal Log-Likelihood + Multinomial Log-Likelihood
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - c_dim_discrete: Number of discrete multinomial dimensions (might be list of independent multinomial spaces).
    - c_dim_continuous: Number of continuous normal dimensions.
    - lambda_z: Weight for the reconstruction loss for the latent z dimensions.

References
----------
.. [1] https://dl.acm.org/doi/10.5555/3157096.3157340
"""

import torch

import numpy as np
import torch.nn as nn

from vegans.utils.layers import LayerReshape
from vegans.models.unconditional.InfoGAN import InfoGAN
from vegans.utils.networks import Generator, Adversary, Encoder
from vegans.utils.utils import get_input_dim, NormalNegativeLogLikelihood
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalInfoGAN(AbstractConditionalGenerativeModel, InfoGAN):
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
    c_dim_discrete: int, list
        Number of discrete multinomial dimensions (might be list of independent multinomial spaces).
    c_dim_continuous: int
        Number of continuous normal dimensions.
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
    lambda_z: float
        Weight for the reconstruction loss for the latent z dimensions.
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
            c_dim_discrete,
            c_dim_continuous,
            optim=None,
            optim_kwargs=None,
            lambda_z=10,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./veganModels/cInfoGAN",
            secure=True):

        c_dim_discrete = [c_dim_discrete] if isinstance(c_dim_discrete, int) else c_dim_discrete
        assert c_dim_discrete == [0] or 0 not in c_dim_discrete, (
            "`c_dim_discrete` has multiple elements. Zero not allowed. Given: {}.".format(c_dim_discrete)
        )
        assert isinstance(c_dim_continuous, int), (
            "`c_dim_continuous` must be of type int. Given: {}.".format(type(c_dim_continuous))
        )
        self.c_dim_discrete = tuple([i for i in list(c_dim_discrete)])
        self.c_dim_continuous = tuple([c_dim_continuous])
        self.c_dim = tuple([sum(self.c_dim_discrete) + sum(self.c_dim_continuous)])
        if len(y_dim) == 3:
            intermediate_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
            gen_in_dim = get_input_dim(dim1=intermediate_in_dim, dim2=self.c_dim)
        else:
            gen_in_dim = get_input_dim(dim1=z_dim, dim2=sum(self.c_dim)+sum(y_dim))
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)

        if secure:
            AbstractConditionalGenerativeModel._check_conditional_network_input(generator, in_dim=z_dim, y_dim=self.c_dim, name="Generator")
        self.generator = Generator(generator, input_size=gen_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversary = Adversary(adversary, input_size=adv_in_dim, adv_type="Discriminator", device=device, ngpu=ngpu, secure=secure)
        self.encoder = Encoder(encoder, input_size=adv_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.neural_nets = {
            "Generator": self.generator, "Adversary": self.adversary, "Encoder": self.encoder
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
        )
        if self.c_dim_discrete != (0,):
            self.multinomial = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.encoder.output_size), np.sum(self.c_dim_discrete)),
                nn.Sigmoid()
            ).to(self.device)

        if self.c_dim_continuous != (0,):
            self.mu = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.encoder.output_size), np.sum(self.c_dim_continuous)),
                LayerReshape(self.c_dim_continuous)
            ).to(self.device)
            self.log_variance = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.encoder.output_size), np.sum(self.c_dim_continuous)),
                nn.ReLU(),
                LayerReshape(self.c_dim_continuous)
            ).to(self.device)

        self.lambda_z = lambda_z
        self.hyperparameters["lambda_z"] = lambda_z
        if self.secure:
            assert (self.generator.output_size == self.x_dim), (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )
            # TODO: Remove those lines or use them again, but not commented
            # if self.encoder.output_size == self.c_dim:
            #     raise ValueError(
            #         "Encoder output size is equal to c_dim, but for InfoGAN the encoder last layers for mu, sigma and discrete values " +
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
            return InfoGAN.encode(self, x=x)

        inpt = self.concatenate(x, y).float()
        return InfoGAN.encode(self, x=inpt)

    def generate(self, y, c=None, z=None):
        """ Generate output with generator / decoder.

        Parameters
        ----------
        z : None, optional
            Latent input vector to produce an output from.
        n : None, optional
            Number of outputs to be generated.

        Returns
        -------
        np.array
            Output produced by generator / decoder.
        """
        if c is None:
            n = len(y)
            c = self.sample_c(n=n)
        if z is None:
            n = len(y)
            z = self.sample(n=n)
        y = self.concatenate(tensor1=y, tensor2=c)
        return self(y=y, z=z)

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
        c = self.sample_c(n=len(Z_batch))
        fake_images = self.generate(y=y_batch, z=Z_batch, c=c)
        fake_concat = self.concatenate(fake_images, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return InfoGAN._calculate_generator_loss(self, X_batch=real_concat, Z_batch=None, fake_images=fake_concat, c=c)

    def _calculate_encoder_loss(self, X_batch, Z_batch, y_batch):
        c = self.sample_c(n=len(Z_batch))
        fake_images = self.generate(y=y_batch, z=Z_batch, c=c).detach()
        fake_concat = self.concatenate(fake_images, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return InfoGAN._calculate_encoder_loss(self, X_batch=real_concat, Z_batch=None, fake_images=fake_concat, c=c)

    def _calculate_adversary_loss(self, X_batch, Z_batch, y_batch):
        c = self.sample_c(n=len(Z_batch))
        fake_images = self.generate(y=y_batch, z=Z_batch, c=c).detach()
        fake_concat = self.concatenate(fake_images, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return InfoGAN._calculate_adversary_loss(self, X_batch=real_concat, Z_batch=None, fake_images=fake_concat)
