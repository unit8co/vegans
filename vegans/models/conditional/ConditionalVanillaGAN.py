"""
ConditionalVanillaGAN
---------------------
Implements the conditional variant of the original Generative adversarial network[1].

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
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1


class ConditionalVanillaGAN(AbstractConditionalGAN1v1):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./CVanillaGAN",
            ngpu=None):

        super().__init__(
            generator=generator, adversariat=adversariat,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss()}
