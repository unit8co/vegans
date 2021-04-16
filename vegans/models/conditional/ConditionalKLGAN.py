"""
ConditionalKLGAN
----------------
Implements the conditional variant of the  Kullback Leibler GAN.

Uses the Kullback Leibler loss for the generator.

Losses:
    - Generator: Kullback-Leibler
    - Autoencoder: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam
"""

import torch

from torch.nn import MSELoss
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1

class ConditionalKLGAN(AbstractConditionalGAN1v1):
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
            eps=1e-5,
            fixed_noise_size=32,
            device=None,
            folder="./LSGAN",
            ngpu=None):

        super().__init__(
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )
        self.eps = eps
        self.hyperparameters["eps"] = eps

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Adversariat": torch.nn.BCELoss()}

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        fake_logits = torch.log(fake_predictions / (1 + self.eps - fake_predictions) + self.eps)

        gen_loss = -torch.mean(fake_logits)
        self._losses.update({"Generator": gen_loss})
