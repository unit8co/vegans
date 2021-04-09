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
            adversariat,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_l1 = 10,
            fixed_noise_size=32,
            device=None,
            folder="./ConditionalPix2Pix",
            ngpu=None):

        super().__init__(
            generator=generator, adversariat=adversariat,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )
        self.lambda_l1 = 10
        self.hyperparameters["lambda_l1"] = self.lambda_l1

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss(), "L1": L1Loss()}

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        gen_loss_original = self.loss_functions["Generator"](
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        gen_loss_pixel_wise = self.loss_functions["L1"](
            X_batch, fake_images
        )
        gen_loss = gen_loss_original + self.lambda_l1*gen_loss_pixel_wise
        self._losses.update({
            "Generator": gen_loss,
            "Generator_Original": gen_loss_original,
            "Generator_L1": gen_loss_pixel_wise
        })