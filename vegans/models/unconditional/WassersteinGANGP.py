"""
WassersteinGANGP
----------------
Implements the Wasserstein GAN Gradient Penalized[1].

Uses the Wasserstein loss to determine the realness of real and fake images.
The Wasserstein loss has several theoretical advantages over the Jensen-Shanon divergence
minimised by the original GAN. In this architecture the critic (discriminator) is often
trained multiple times for every generator step.
Lipschitz continuity is "enforced" by gradient penalization.

Losses:
    - Generator: Wasserstein
    - Critic: Wasserstein + Gradient penalization
Default optimizer:
    - torch.optim.RMSprop
Custom parameter:
    - lambda_grad: Weight for the reconstruction loss of the gradients. Pushes the norm of the gradients to 1.

References
----------
.. [1] https://arxiv.org/abs/1704.00028
"""

import torch

import numpy as np

from vegans.utils.utils import WassersteinLoss
from vegans.models.unconditional.AbstractGAN1v1 import AbstractGAN1v1


class WassersteinGANGP(AbstractGAN1v1):
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
    lambda_grad: float
        Weight for the reconstruction loss of the gradients. Pushes the norm of the gradients to 1.
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
            lmbda_grad=10,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./WassersteinGANGP",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            z_dim=z_dim, x_dim=x_dim, adv_type="Critic",
            optim=optim, optim_kwargs=optim_kwargs,
            feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device,
            folder=folder,
            ngpu=ngpu, secure=secure
        )
        self.lmbda_grad = lmbda_grad
        self.hyperparameters["lmbda_grad"] = lmbda_grad

    def _default_optimizer(self):
        return torch.optim.RMSprop

    def _define_loss(self):
        loss_functions = {"Generator": WassersteinLoss(), "Adversary": WassersteinLoss(), "GP": self._gradient_penalty}
        return loss_functions

    def _gradient_penalty(self, real_samples, fake_samples):
        sample_shape = (real_samples.size(0), *[1 for _ in range(len(real_samples.shape)-1)])
        alpha = torch.rand(size=sample_shape, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
        d_interpolates = self.adversary(interpolates).to(self.device)
        dummy = torch.ones_like(d_interpolates, requires_grad=False).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=dummy,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    #########################################################################
    # Actions during training
    #########################################################################
    def _calculate_adversary_loss(self, X_batch, Z_batch, fake_images=None):
        if fake_images is None:
            fake_images = self.generate(z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images)
        real_predictions = self.predict(x=X_batch)

        adv_loss_fake = self.loss_functions["Adversary"](
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss_grad = self.loss_functions["GP"](X_batch, fake_images)
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real) + self.lmbda_grad*adv_loss_grad
        return {
            "Adversary": adv_loss,
            "Adversary_fake": adv_loss_fake,
            "Adversary_real": adv_loss_real,
            "Adversary_grad": adv_loss_grad,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        }