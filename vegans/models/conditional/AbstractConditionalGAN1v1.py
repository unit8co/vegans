import os
import torch

import numpy as np

from vegans.utils.utils import get_input_dim
from vegans.utils.networks import Generator, Adversariat
from vegans.models.unconditional.AbstractGAN1v1 import AbstractGAN1v1
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel


class AbstractConditionalGAN1v1(AbstractConditionalGenerativeModel, AbstractGAN1v1):
    """ Special half abstract class for conditional GAN with structure of one generator and
    one discriminator / critic. Examples are the original `ConditionalVanillaGAN`,
    `ConditionalWassersteinGAN` and `ConditionalWassersteinGANGP`.
    """

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
            adv_type,
            optim=None,
            optim_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./AbstractGAN1v1",
            ngpu=0):

        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        AbstractGAN1v1.__init__(
            self, generator=generator, adversariat=adversariat, x_dim=adv_in_dim, z_dim=gen_in_dim,
            adv_type=adv_type, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=0,
            _called_from_conditional=True
        )
        AbstractConditionalGenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        assert (self.generator.output_size == self.x_dim), (
            "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
        )


    #########################################################################
    # Actions during training
    #########################################################################
    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        gen_loss = self.loss_functions["Generator"](
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self._losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        real_predictions = self.predict(x=X_batch, y=y_batch)

        adv_loss_fake = self.loss_functions["Adversariat"](
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversariat"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        })