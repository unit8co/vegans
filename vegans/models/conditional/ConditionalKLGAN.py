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
Custom parameter:
    - eps: Small value preventing overflow and nans when calculating the Kullback-Leibler divergence.

References
----------
.. [1] https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
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
            adversary,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            eps=1e-5,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./CKLGAN",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu, secure=secure
        )
        self.eps = eps
        self.hyperparameters["eps"] = eps

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        loss_functions = {"Adversary": torch.nn.BCELoss()}
        return loss_functions

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images, y=y_batch)
            fake_logits = torch.log(fake_predictions / (1 + self.eps - fake_predictions) + self.eps)
            gen_loss = -torch.mean(fake_logits)
        else:
            gen_loss = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images, y_batch=y_batch)

        return {"Generator": gen_loss}
