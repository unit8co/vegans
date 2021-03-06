"""
ConditionalPix2Pix
------------------
Implements the Pix2Pix GAN[1].

Uses the binary cross-entropy norm for evaluating the realness of real and fake images.
Also enforces a L1 pixel wise penalty on the generated images.

Losses:
    - Generator: Binary cross-entropy + L1 (Mean Absolute Error)
    - Discriminator: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_x: Weight for the reconstruction loss for the real x dimensions.
References
----------
.. [1] https://arxiv.org/abs/1611.07004
"""

import torch

from torch.nn import BCELoss, L1Loss
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1

class ConditionalPix2Pix(AbstractConditionalGAN1v1):
    """
    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
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
    lambda_x: float
        Weight for the reconstruction loss of the real x dimensions.
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
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_x=10,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./veganModels/cPix2Pix",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )
        self.lambda_x = 10
        self.hyperparameters["lambda_x"] = self.lambda_x

    def _define_loss(self):
        loss_functions = {"Generator": BCELoss(), "Adversary": BCELoss(), "L1": L1Loss()}
        return loss_functions

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images, y=y_batch)
            gen_loss_original = self.loss_functions["Generator"](
                fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
            )
        else:
            fake_concat = self.concatenate(fake_images, y_batch)
            real_concat = self.concatenate(X_batch, y_batch)
            gen_loss_original = self._calculate_feature_loss(X_real=real_concat, X_fake=fake_concat)
        gen_loss_pixel_wise = self.loss_functions["L1"](
            X_batch, fake_images
        )
        gen_loss = gen_loss_original + self.lambda_x*gen_loss_pixel_wise
        return {
            "Generator": gen_loss,
            "Generator_Original": gen_loss_original,
            "Generator_L1": gen_loss_pixel_wise
        }