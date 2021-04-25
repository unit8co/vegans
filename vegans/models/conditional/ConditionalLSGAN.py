"""
ConditionalLSGAN
----------------
Implements the conditional variant of the Least-Squares GAN[1].

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
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1

class ConditionalLSGAN(AbstractConditionalGAN1v1):
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
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./CLSGAN",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu, secure=secure
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        loss_functions = {"Generator": torch.nn.MSELoss(), "Adversary": torch.nn.MSELoss()}
        return loss_functions