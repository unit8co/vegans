"""
EBGAN
-----
Implements the Energy based GAN[1].

Uses an auto-encoder as the adversary structure.

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
    """
    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
    x_dim : list, tuple
        Number of the output dimensions of the generator and input dimension of the discriminator / critic.
        In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
    z_dim : int, list, tuple
        Number of the latent dimensions for the generator input. Might have dimensions of an image.
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
    m: float
        Cut off for the hinge loss. Look at reference for more information.
    feature_layer : torch.nn.*
        Output layer used to compute the feature loss. Should be from either the discriminator or critic.
        If `feature_layer` is not None, the original generator loss is replaced by a feature loss, introduced
        [here](https://arxiv.org/abs/1606.03498v1).
    fixed_noise_size : int
        Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
        subdirectory, the tensorboard images tab and the samples in get_training_results().
    device : string
        Device used while training the model. Either "cpu" or "cuda".
    ngpu : int
        Number of gpus used during training if device == "cuda".
    folder : string
        Creates a folder in the current working directory with this name. All relevant files like summary, images, models and
        tensorboard output are written there. Existing folders are never overwritten or deleted. If a folder with the same name
        already exists a time stamp is appended to make it unique.
    """

    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversary,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            m=None,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./EBGAN",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            z_dim=z_dim, x_dim=x_dim, adv_type="Autoencoder",
            optim=optim, optim_kwargs=optim_kwargs,
            feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu, secure=secure
        )

        if self.secure:
            assert self.adversary.output_size == x_dim, (
                "AutoEncoder structure used for adversary. Output dimensions must equal x_dim. " +
                "Output: {}. x_dim: {}.".format(self.adversary.output_size, x_dim)
            )
        self.m = m
        self.hyperparameters["m"] = m

    def _define_loss(self):
        loss_functions = {"Generator": torch.nn.MSELoss(), "Adversary": torch.nn.MSELoss()}
        return loss_functions

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
                fake_predictions, fake_images
            )
        else:
            gen_loss = self._calculate_feature_loss(X_real=X_batch, X_fake=fake_images)
        return {"Generator": gen_loss}

    def _calculate_adversary_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images)
        real_predictions = self.predict(x=X_batch)

        adv_loss_fake = self.loss_functions["Adversary"](
            fake_predictions, fake_images
        )
        if adv_loss_fake < self.m:
            adv_loss_fake = self.m - adv_loss_fake
        else:
            adv_loss_fake = torch.Tensor([0]).to(self.device)
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictions, X_batch
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real).float()
        return {
            "Adversary": adv_loss,
            "Adversary_fake": adv_loss_fake,
            "Adversary_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        }
