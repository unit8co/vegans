"""
ConditionalEBGAN
----------------
Implements conditional variant of the Energy based GAN[1].

Uses an auto-encoder as the adversariat structure.

Losses:
    - Generator: L2 (Mean Squared Error)
    - Autoencoder: L2 (Mean Squared Error)
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://arxiv.org/pdf/1609.03126.pdf
"""

import torch

from torch.nn import MSELoss
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1

class ConditionalEBGAN(AbstractConditionalGAN1v1):
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
            m,
            optim=None,
            optim_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./LSGAN",
            ngpu=None):

        super().__init__(
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, adv_type="AutoEncoder",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )
        assert self.adversariat.output_size == x_dim, (
            "AutoEncoder structure used for adversariat. Output dimensions must equal x_dim. " +
            "Output: {}. x_dim: {}.".format(self.adversariat.output_size, x_dim)
        )
        self.m = m
        self.hyperparameters["m"] = m

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Generator": torch.nn.MSELoss(), "Adversariat": torch.nn.MSELoss()}

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        gen_loss = self.loss_functions["Generator"](
            fake_images, fake_predictions
        )
        self._losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        real_predictions = self.predict(x=X_batch, y=y_batch)

        adv_loss_fake = self.loss_functions["Adversariat"](
            fake_predictions, fake_images
        )
        if adv_loss_fake < self.m:
            adv_loss_fake = self.m - adv_loss_fake
        else:
            adv_loss_fake = torch.Tensor([0]).to(self.device)
        adv_loss_real = self.loss_functions["Adversariat"](
            real_predictions, X_batch
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real).float()
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        })
