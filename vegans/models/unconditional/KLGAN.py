"""
KLGAN
-----
Implements the Kullback Leibler GAN.

Uses the Kullback Leibler loss for the generator.

Losses:
    - Generator: Kullback-Leibler
    - Autoencoder: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - eps: Small value preventing overflow and nans when calculating the Kullback-Leibler divergence.
"""

import torch

from torch.nn import MSELoss
from vegans.models.unconditional.AbstractGAN1v1 import AbstractGAN1v1

class KLGAN(AbstractGAN1v1):
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
            eps=1e-5,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder="./KLGAN",
            ngpu=None,
            secure=True):

        super().__init__(
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu, secure=secure
        )
        self.eps = eps
        self.hyperparameters["eps"] = eps

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Adversariat": torch.nn.BCELoss()}

    def _calculate_generator_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch)
        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images)
            fake_logits = torch.log(fake_predictions / (1 + self.eps - fake_predictions) + self.eps)
            gen_loss = -torch.mean(fake_logits)
        else:
            gen_loss = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images)

        self._losses.update({"Generator": gen_loss})
