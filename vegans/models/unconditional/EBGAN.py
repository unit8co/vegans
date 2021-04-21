"""
EBGAN
-----
Implements the Energy based GAN[1].

Uses an auto-encoder as the adversariat structure.

Losses:
    - Generator: L2 (Mean Squared Error)
    - Autoencoder: L2 (Mean Squared Error)
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - m: Cut off for the hinge loss. Look at reference for more information.

References
----------
.. [1] https://arxiv.org/pdf/1609.03126.pdf
"""

import torch

from torch.nn import MSELoss
from vegans.models.unconditional.AbstractGAN1v1 import AbstractGAN1v1

class EBGAN(AbstractGAN1v1):
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
            m=None,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder="./EBGAN",
            ngpu=None,
            secure=True):

        super().__init__(
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, adv_type="AutoEncoder",
            optim=optim, optim_kwargs=optim_kwargs,
            feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu, secure=secure
        )

        if self.secure:
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

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, steps,
        print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard):
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = super()._set_up_training(
            X_train, y_train, X_test, y_test, epochs, batch_size, steps,
            print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard
        )
        if self.m is None:
            self.m = np.mean(X_train)
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods

    def _calculate_generator_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch)
        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images)
            gen_loss = self.loss_functions["Generator"](
                fake_images, fake_predictions
            )
        else:
            gen_loss = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images)
        self._losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images)
        real_predictions = self.predict(x=X_batch)

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
