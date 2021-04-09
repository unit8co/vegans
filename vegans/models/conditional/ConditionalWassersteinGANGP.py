"""
ConditionalWassersteinGANGP
---------------------------
Implements the conditional variant of the Wasserstein GAN Gradient Penalized[1].

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

References
----------
.. [1] https://arxiv.org/abs/1704.00028
"""

import torch

import numpy as np

from vegans.utils.utils import wasserstein_loss, concatenate
from vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1


class ConditionalWassersteinGANGP(AbstractConditionalGAN1v1):
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
            fixed_noise_size=32,
            lmbda_grad=10,
            device=None,
            folder="./WassersteinGANGP",
            ngpu=None):

        super().__init__(
            generator=generator, adversariat=adversariat,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device,
            folder=folder,
            ngpu=ngpu
        )
        self.lmbda_grad = lmbda_grad
        self.hyperparameters["lmbda_grad"] = lmbda_grad

    def _default_optimizer(self):
        return torch.optim.RMSprop

    def _define_loss(self):
        self.loss_functions = {"Generator": wasserstein_loss, "Adversariat": wasserstein_loss, "GP": self._gradient_penalty}

    def _gradient_penalty(self, real_images, fake_images):
        alpha = torch.Tensor(np.random.random((real_images.size(0), 1, 1, 1))).to(self.device)
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True).float()
        d_interpolates = self.adversariat(interpolates).to(self.device)
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
        adv_loss_grad = self.loss_functions["GP"](
            concatenate(X_batch, y_batch),
            concatenate(fake_images, y_batch)
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real) + self.lmbda_grad*adv_loss_grad
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
            "Adversariat_grad": adv_loss_grad,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        })