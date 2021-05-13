"""
InfoGAN
-------
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
from torch.nn import CrossEntropyLoss, BCELoss
from vegans.utils.networks import Generator, Adversary, Encoder
from vegans.utils.utils import get_input_dim, concatenate, NormalNegativeLogLikelihood
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class InfoGAN(AbstractGenerativeModel):
    """
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
            c_dim_discrete,
            c_dim_continuous,
            optim=None,
            optim_kwargs=None,
            lambda_z=10,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./InfoGAN",
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
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=self.c_dim)

        if secure:
            AbstractConditionalGenerativeModel._check_conditional_network_input(generator, in_dim=z_dim, y_dim=self.c_dim, name="Generator")
        self.generator = Generator(generator, input_size=gen_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversary = Adversary(adversary, input_size=x_dim, adv_type="Discriminator", device=device, ngpu=ngpu, secure=secure)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu, secure=secure)
        self.neural_nets = {
            "Generator": self.generator, "Adversary": self.adversary, "Encoder": self.encoder
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
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
                LayerReshape(shape=self.c_dim_continuous)
            ).to(self.device)
            self.log_variance = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.encoder.output_size), np.sum(self.c_dim_continuous)),
                nn.ReLU(),
                LayerReshape(shape=self.c_dim_continuous)
            ).to(self.device)

        self.lambda_z = lambda_z
        self.hyperparameters["lambda_z"] = lambda_z
        if self.secure:
            assert (self.generator.output_size == self.x_dim), (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )
            # if self.encoder.output_size == self.c_dim:
            #     raise ValueError(
            #         "Encoder output size is equal to c_dim, but for InfoGAN the encoder last layers for mu, sigma and discrete values " +
            #         "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            #     )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        loss_functions = {
            "Generator": BCELoss(), "Adversary": BCELoss(),
            "Discrete": CrossEntropyLoss(), "Continuous": NormalNegativeLogLikelihood()
        }
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x):
        return self.encoder(x)

    def sample_c(self, n):
        """ Sample the conditional vector.

        Parameters
        ----------
        n : int
            Number of outputs to be generated.
        """
        samples = []
        if self.c_dim_discrete[0] != 0:
            for c in self.c_dim_discrete:
                weights = torch.ones(size=(n, c))
                c_discrete = torch.zeros(size=(n, c), device=self.device)
                idx = torch.multinomial(input=weights, num_samples=1)
                for row in range(n):
                    c_discrete[row, idx[row]] = 1.
                samples.append(c_discrete)

        if self.c_dim_continuous[0] != 0:
            c_continuous = torch.randn(size=(n, *self.c_dim_continuous), requires_grad=True, device=self.device)
            samples.append(c_continuous)

        samples = torch.cat(tuple(samples), axis=1)
        return samples

    def generate(self, c=None, z=None, n=None):
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
            n = len(z) if z is not None else None
            assert n is not None, "If `c=None`, n must be not None."
            c = self.sample_c(n=n)
        if z is None:
            n = len(c) if c is not None else None
            assert n is not None, "If `c=None`, n must be not None."
            z = self.sample(n=n)
        z = concatenate(tensor1=z, tensor2=c)
        return self(z=z)

    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Generator":
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Adversary":
            losses = self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Encoder":
            losses = self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
            losses.update(self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch))
            losses.update(self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch))
        return losses

    def _calculate_generator_loss(self, X_batch, Z_batch):
        c = self.sample_c(n=len(Z_batch))
        fake_images = self.generate(z=Z_batch, c=c)
        encoded = self.encode(x=fake_images)

        if self.c_dim_discrete[0] != 0:
            reconstructed_c_discrete = self.multinomial(encoded)
        if self.c_dim_continuous[0] != 0:
            reconstructed_mu = self.mu(encoded)
            reconstructed_variance = self.log_variance(encoded).exp()

        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images)
            gen_loss_original = self.loss_functions["Generator"](
                fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
            )
        else:
            gen_loss_original = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images)
        discrete_encoder_loss = torch.Tensor([0]).to(self.device)
        start = 0
        if self.c_dim_discrete[0] != 0:
            for c_dim in self.c_dim_discrete:
                end = start + c_dim
                discrete_encoder_loss += self.loss_functions["Discrete"](
                    reconstructed_c_discrete[:, start:end], torch.argmax(c[:, start:end].long(), axis=1)
                )
                start += c_dim
        if self.c_dim_continuous[0] != 0:
            continuous_encoder_loss = self.loss_functions["Continuous"](
                x=c[:, -self.c_dim_continuous[0]:], mu=reconstructed_mu, variance=reconstructed_variance
            )
        else:
            continuous_encoder_loss = torch.Tensor([0]).to(self.device)

        gen_loss = gen_loss_original + self.lambda_z*(discrete_encoder_loss + continuous_encoder_loss)
        return {
            "Generator": gen_loss,
            "Generator_Original": gen_loss_original,
            "Generator_Discrete": self.lambda_z*discrete_encoder_loss,
            "Generator_Continuous": self.lambda_z*continuous_encoder_loss
        }

    def _calculate_encoder_loss(self, X_batch, Z_batch):
        c = self.sample_c(n=len(Z_batch))
        fake_images = self.generate(z=Z_batch, c=c).detach()
        encoded = self.encode(x=fake_images)

        if self.c_dim_discrete[0] != 0:
            reconstructed_c_discrete = self.multinomial(encoded)
        if self.c_dim_continuous[0] != 0:
            reconstructed_mu = self.mu(encoded)
            reconstructed_variance = self.log_variance(encoded).exp()

        discrete_encoder_loss = torch.Tensor([0]).to(self.device)
        start = 0
        if self.c_dim_discrete[0] != 0:
            for c_dim in self.c_dim_discrete:
                end = start + c_dim
                discrete_encoder_loss += self.loss_functions["Discrete"](
                    reconstructed_c_discrete[:, start:end], torch.argmax(c[:, start:end].long(), axis=1)
                )
                start += c_dim
        if self.c_dim_continuous[0] != 0:
            continuous_encoder_loss = self.loss_functions["Continuous"](
                c[:, -self.c_dim_continuous[0]:], reconstructed_mu, reconstructed_variance
            )
        else:
            continuous_encoder_loss = torch.Tensor([0]).to(self.device)

        enc_loss = 0.5*(discrete_encoder_loss + continuous_encoder_loss)
        return {
            "Encoder": enc_loss,
            "Encoder_Discrete": discrete_encoder_loss,
            "Encoder_Continuous": continuous_encoder_loss
        }

    def _calculate_adversary_loss(self, X_batch, Z_batch):
        c = self.sample_c(n=len(Z_batch))
        fake_images = self.generate(z=Z_batch, c=c).detach()
        fake_predictions = self.predict(x=fake_images)
        real_predictions = self.predict(x=X_batch)

        adv_loss_fake = self.loss_functions["Adversary"](
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        return {
            "Adversary": adv_loss,
            "Adversary_fake": adv_loss_fake,
            "Adversary_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        }