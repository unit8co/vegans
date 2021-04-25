"""
ConditionalPix2Pix
------------------
Implements the Pix2Pix GAN[1].

Uses the binary cross-entropy norm for evaluating the realness of real and fake images.
Also enforces a L1 pixel wise penalty on the generated images.

Losses:
    - Generator: Binary cross-entropy + L1 (Mean Absolute Error)
    - Discriminator: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_x: Weight for the reconstruction loss for the real x dimensions.
References
----------
.. [1] https://arxiv.org/abs/1611.07004
"""

import torch

from torch.nn import BCELoss, L1Loss
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1

class ConditionalPix2Pix(AbstractConditionalGAN1v1):
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
            lambda_x=10,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./CPix2Pix",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )
        self.lambda_x = 10
        self.hyperparameters["lambda_x"] = self.lambda_x

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        loss_functions = {"Generator": BCELoss(), "Adversary": BCELoss(), "L1": L1Loss()}
        return loss_functions

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images, y=y_batch)
            gen_loss_original = self.loss_functions["Generator"](
                fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
            )
        else:
            gen_loss_original = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images, y_batch=y_batch)
        gen_loss_pixel_wise = self.loss_functions["L1"](
            X_batch, fake_images
        )
        gen_loss = gen_loss_original + self.lambda_x*gen_loss_pixel_wise
        return {
            "Generator": gen_loss,
            "Generator_Original": gen_loss_original,
            "Generator_L1": gen_loss_pixel_wise
        }