"""
LRGAN
-----
Implements the latent regressor GAN well described in the BicycleGAN paper[1].

It introduces an encoder network which maps the generator output back to the latent
input space. This should help to prevent mode collapse and improve image variety.

Losses:
    - Generator: Binary cross-entropy + L1-latent-loss (Mean Absolute Error)
    - Discriminator: Binary cross-entropy
    - Encoder: L1-latent-loss (Mean Absolute Error)
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_z: Weight for the reconstruction loss for the latent z dimensions.

References
----------
.. [1] https://arxiv.org/pdf/1711.11586.pdf
"""

import torch

from torch.nn import L1Loss
from vegans.utils.networks import Generator, Adversary, Encoder
from vegans.models.unconditional.AbstractGANGAE import AbstractGANGAE

class LRGAN(AbstractGANGAE):
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
        Weight for the reconstruction loss for the latent z dimensions.
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
            folder="./LRGAN",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary, encoder=encoder,
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, adv_type=adv_type, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, ngpu=ngpu, folder=folder, secure=secure
        )
        self.lambda_z = lambda_z
        self.hyperparameters["lambda_z"] = lambda_z

        if self.secure:
            assert self.encoder.output_size == self.z_dim, (
                "Encoder output shape must be equal to z_dim. {} vs. {}.".format(self.encoder.output_size, self.z_dim)
            )

    def _define_loss(self):
        loss_functions = super()._define_loss()
        loss_functions.update({"L1": L1Loss()})
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def _calculate_generator_loss(self, X_batch, Z_batch, fake_images=None):
        if fake_images is None:
            fake_images = self.generate(z=Z_batch)
        fake_Z = self.encode(x=fake_images)

        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images)
            gen_loss_original = self.loss_functions["Generator"](
                fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
            )
        else:
            gen_loss_original = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images)
        latent_space_regression = self.loss_functions["L1"](
            fake_Z, Z_batch
        )
        gen_loss = gen_loss_original + self.lambda_z*latent_space_regression
        return {
            "Generator": gen_loss,
            "Generator_Original": gen_loss_original,
            "Generator_L1": self.lambda_z*latent_space_regression
        }

    def _calculate_encoder_loss(self, X_batch, Z_batch, fake_images=None):
        if fake_images is None:
            fake_images = self.generate(z=Z_batch).detach()
        fake_Z = self.encode(x=fake_images)
        latent_space_regression = self.loss_functions["L1"](
            fake_Z, Z_batch
        )
        return {
            "Encoder": latent_space_regression
        }

    def _calculate_adversary_loss(self, X_batch, Z_batch, fake_images=None):
        if fake_images is None:
            fake_images = self.generate(z=Z_batch).detach()
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
