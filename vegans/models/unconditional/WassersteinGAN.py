"""
WassersteinGAN
--------------
Implements the Wasserstein GAN[1].

Uses the Wasserstein loss to determine the realness of real and fake images.
The Wasserstein loss has several theoretical advantages over the Jensen-Shanon divergence
minimised by the original GAN. In this architecture the critic (discriminator) is often
trained multiple times for every generator step.
Lipschitz continuity is "enforced" by weight clipping.

Losses:
    - Generator: Wasserstein
    - Critic: Wasserstein
Default optimizer:
    - torch.optim.RMSprop
Custom parameter:
    - clip_val: Clip value for the critic to maintain Lipschitz continuity.

References
----------
.. [1] https://export.arxiv.org/pdf/1701.07875
"""

import torch

import numpy as np

from vegans.models.unconditional.AbstractGAN1v1 import AbstractGAN1v1
from vegans.utils.utils import WassersteinLoss


class WassersteinGAN(AbstractGAN1v1):
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
            clip_val=0.01,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder="./WassersteinGAN",
            ngpu=None,
            secure=True):

        super().__init__(
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, adv_type="Critic",
            optim=optim, optim_kwargs=optim_kwargs,
            feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device,
            folder=folder,
            ngpu=ngpu, secure=secure
        )
        self._clip_val = clip_val
        self.hyperparameters["clip_val"] = clip_val

    def _default_optimizer(self):
        return torch.optim.RMSprop

    def _default_optimizer(self):
        return torch.optim.RMSprop

    def _define_loss(self):
        self.loss_functions = {"Generator": WassersteinLoss(), "Adversariat": WassersteinLoss()}


    #########################################################################
    # Actions during training
    #########################################################################
    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversariat":
                for p in self.adversariat.parameters():
                    p.data.clamp_(-self._clip_val, self._clip_val)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]