"""
ConditionalWassersteinGAN
--------------
Implements the conditional variant of the Wasserstein GAN[1].

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

References
----------
.. [1] https://export.arxiv.org/pdf/1701.07875
"""

import torch

import numpy as np

from vegans.utils.utils import wasserstein_loss
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1


class ConditionalWassersteinGAN(AbstractConditionalGAN1v1):
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
            clip_val=0.01,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder="./CWassersteinGAN",
            ngpu=None):

        super().__init__(
            generator=generator, adversariat=adversariat,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic",
            optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device,
            folder=folder,
            ngpu=ngpu
        )
        self._clip_val = clip_val
        self.hyperparameters["clip_val"] = clip_val

    def _default_optimizer(self):
        return torch.optim.RMSprop

    def _define_loss(self):
        self.loss_functions = {"Generator": wasserstein_loss, "Adversariat": wasserstein_loss}


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