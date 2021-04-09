"""
LSGAN
-----
Implements the Least-Squares GAN[1].

Uses the L2 norm for evaluating the realness of real and fake images.

Losses:
    - Generator: L2 (Mean Squared Error)
    - Discriminator: L2 (Mean Squared Error)
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf
"""

import torch

from torch.nn import MSELoss
from vegans.models.unconditional.AbstractGAN1v1 import AbstractGAN1v1

class LSGAN(AbstractGAN1v1):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./VanillaGAN",
            ngpu=None):

        super().__init__(
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Generator": torch.nn.MSELoss(), "Adversariat": torch.nn.MSELoss()}