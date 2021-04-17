import os
import torch

import numpy as np

from vegans.utils.networks import Generator, Adversariat
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel


class AbstractGAN1v1(AbstractGenerativeModel):
    """ Special half abstract class for GAN with structure of one generator and
    one discriminator / critic. Examples are the original `VanillaGAN`, `WassersteinGAN`
    and `WassersteinGANGP`.
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
            adv_type,
            optim=None,
            optim_kwargs=None,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder=None,
            ngpu=0,
            _called_from_conditional=False):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu)
        self.adversariat = Adversariat(adversariat, input_size=x_dim, adv_type=adv_type, device=device, ngpu=ngpu)
        self.neural_nets = {"Generator": self.generator, "Adversariat": self.adversariat}

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        if not _called_from_conditional:
            assert (self.generator.output_size == self.x_dim), (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )


    #########################################################################
    # Actions during training
    #########################################################################
    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)

    def _calculate_generator_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch)
        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images)
            gen_loss = self.loss_functions["Generator"](
                fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
            )
        else:
            gen_loss = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images)
        self._losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images)
        real_predictions = self.predict(x=X_batch)

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