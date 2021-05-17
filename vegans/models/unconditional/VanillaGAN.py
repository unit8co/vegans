"""
VanillaGAN
----------
Implements the original Generative adversarial network[1].

Uses the binary cross-entropy for evaluating the realness of real and fake images.
The discriminator tries to output 1 for real images and 0 for fake images, whereas the
generator tries to force the discriminator to output 1 for fake images.

Losses:
    - Generator: Binary cross-entropy
    - Discriminator: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
"""

import torch

from torch.nn import BCELoss
from vegans.models.unconditional.AbstractGAN1v1 import AbstractGAN1v1


class VanillaGAN(AbstractGAN1v1):
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
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
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
            optim=None,
            optim_kwargs=None,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./VanillaGAN",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            z_dim=z_dim, x_dim=x_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu, secure=secure
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        loss_functions = {"Generator": BCELoss(), "Adversary": BCELoss()}
        return loss_functions
