"""
LRGAN
-----
Implements the conditional variant of the latent regressor GAN well described in the BicycleGAN paper[1].

It introduces an encoder network which maps the generator output back to the latent
input space. This should help to prevent mode collapse and improve image variety.

Losses:
    - Generator: Binary cross-entropy + L1-latent-loss (Mean Absolute Error)
    - Discriminator: Binary cross-entropy
    - Encoder: L1-latent-loss (Mean Absolute Error)
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://arxiv.org/pdf/1711.11586.pdf
"""

import torch

from torch.nn import BCELoss, L1Loss
from torch.nn import MSELoss as L2Loss

from vegans.utils.utils import get_input_dim
from vegans.utils.utils import wasserstein_loss
from vegans.utils.networks import Generator, Adversariat, Encoder
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalLRGAN(AbstractConditionalGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            encoder,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_L1=10,
            adv_type="Discriminator",
            fixed_noise_size=32,
            device=None,
            folder="./CLRGAN",
            ngpu=0):

        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        adv_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        AbstractConditionalGenerativeModel._check_conditional_network_input(encoder, in_dim=x_dim, y_dim=y_dim, name="Encoder")
        AbstractConditionalGenerativeModel._check_conditional_network_input(generator, in_dim=z_dim, y_dim=y_dim, name="Generator")
        AbstractConditionalGenerativeModel._check_conditional_network_input(adversariat, in_dim=z_dim, y_dim=y_dim, name="Adversariat")
        self.adv_type = adv_type
        self.encoder = Encoder(encoder, input_size=enc_in_dim, device=device, ngpu=ngpu)
        self.generator = Generator(generator, input_size=gen_in_dim, device=device, ngpu=ngpu)
        self.adversariat = Adversariat(adversariat, input_size=adv_in_dim, adv_type=adv_type, device=device, ngpu=ngpu)
        self.neural_nets = {
            "Generator": self.generator, "Adversariat": self.adversariat, "Encoder": self.encoder
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        self.lambda_L1 = lambda_L1
        self.hyperparameters["lambda_L1"] = lambda_L1
        assert (self.generator.output_size == self.x_dim), (
            "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
        )
        assert (self.encoder.output_size == self.z_dim), (
            "Encoder output shape must be equal to z_dim. {} vs. {}.".format(self.encoder.output_size, self.z_dim)
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            sself.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss(), "L1": L1Loss()}
        elif self.adv_type == "Critic":
            self.loss_functions = {"Generator": wasserstein_loss, "Adversariat": wasserstein_loss, "L1": L1Loss()}
        else:
            raise NotImplementedError("'adv_type' must be one of Discriminator or Critic.")


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x, y):
        inpt = self.concatenate(x, y).float()
        return self.encoder(inpt)

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Encoder":
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        encoded_space = self.encode(x=fake_images, y=y_batch)

        gen_loss_original = self.loss_functions["Generator"](
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        latent_space_regression = self.loss_functions["L1"](
            encoded_space, Z_batch
        )
        gen_loss = gen_loss_original + self.lambda_L1*latent_space_regression
        self._losses.update({
            "Generator": gen_loss,
            "Generator_Original": gen_loss_original,
            "Generator_L1": self.lambda_L1*latent_space_regression
        })

    def _calculate_encoder_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch).detach()
        encoded_space = self.encoder(fake_images)
        latent_space_regression = self.loss_functions["L1"](
            encoded_space, Z_batch
        )
        self._losses.update({
            "Encoder": latent_space_regression
        })

    def _calculate_adversariat_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        real_predictions = self.predict(x=X_batch.float(), y=y_batch)

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

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversariat":
                if self.adv_type == "Critic":
                    for p in self.adversariat.parameters():
                        p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]