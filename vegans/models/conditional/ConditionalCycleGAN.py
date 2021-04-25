"""
CycleGAN
--------
Implements the CycleGAN[1], a method for unpaired image to image translation tasks.



Losses:
    - Generator1 and Generator2:
    - Discriminator1 and Discriminator2:
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_x: Weight for the reconstruction loss of the real x dimensions.

References
----------
.. [1] https://arxiv.org/abs/1703.10593
"""

import torch

import numpy as np
import torch.nn as nn

from torch.nn import MSELoss
from torch.nn import BCELoss, L1Loss

from vegans.utils.utils import get_input_dim
from vegans.utils.utils import WassersteinLoss
from vegans.utils.networks import Generator, Adversary, Autoencoder
from vegans.models.conditional.AbstractConditionalGenerativeModel import AbstractConditionalGenerativeModel

class ConditionalCycleGAN(AbstractConditionalGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generatorX_Y,
            adversaryX_Y,
            generatorY_X,
            adversaryY_X,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            lambda_x=10,
            adv_type="Discriminator",
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./CCycleGAN",
            secure=True):

        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        if secure:
            assert x_dim == y_dim, (
                "`x_dim` and `y_dim` must be equal in the current implementation of CycleGAN. Given: {} and {}.".format(x_dim, y_dim)
            )
            AbstractConditionalGenerativeModel._check_conditional_network_input(generatorX_Y, in_dim=z_dim, y_dim=x_dim, name="GeneratorX_Y")
            AbstractConditionalGenerativeModel._check_conditional_network_input(adversaryX_Y, in_dim=y_dim, y_dim=x_dim, name="AdversaryX_Y")
            AbstractConditionalGenerativeModel._check_conditional_network_input(generatorY_X, in_dim=z_dim, y_dim=y_dim, name="GeneratorY_X")
            AbstractConditionalGenerativeModel._check_conditional_network_input(adversaryY_X, in_dim=x_dim, y_dim=y_dim, name="AdversaryY_X")
        self.adv_type = adv_type
        self.generatorX_Y = Generator(generatorX_Y, input_size=gen_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversaryX_Y = Adversary(adversaryX_Y, input_size=adv_in_dim, adv_type=adv_type, device=device, ngpu=ngpu, secure=secure)
        self.generatorY_X = Generator(generatorY_X, input_size=gen_in_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversaryY_X = Adversary(adversaryY_X, input_size=adv_in_dim, adv_type=adv_type, device=device, ngpu=ngpu, secure=secure)
        self.autoencoder = Autoencoder(encoder=self.generatorX_Y, decoder=self.generatorY_X)
        self.neural_nets = {
            "Autoencoder": self.autoencoder, "AdversaryX_Y": self.adversaryX_Y, "AdversaryY_X": self.adversaryY_X
        }

        self.generator = self.generatorX_Y
        self.adversary = self.adversaryX_Y
        super().__init__(
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=None,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
        )

        self.lambda_x = lambda_x
        self.hyperparameters["lambda_x"] = lambda_x

        if self.secure:
            assert (self.generatorX_Y.output_size == self.x_dim), (
                "GeneratorX_Y output shape must be equal to x_dim. {} vs. {}.".format(self.generatorX_Y.output_size, self.x_dim)
            )
            assert (self.generatorY_X.output_size == self.x_dim), (
                "GeneratorY_X output shape must be equal to x_dim. {} vs. {}.".format(self.generatorY_X.output_size, self.x_dim)
            )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            loss_functions = {"Reconstruction": MSELoss(), "Adversary": MSELoss()}
        elif self.adv_type == "Critic":
            loss_functions = {"Reconstruction": MSELoss(), "Adversary": WassersteinLoss()}
        else:
            raise NotImplementedError("'adv_type' must be one of Discriminator or Critic.")
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def generate(self, y, z=None, who="GeneratorX_Y"):
        """ Generate output with generator.

        Parameters
        ----------
        y : np.array
            Labels for outputs to be produced.
        z : None, optional
            Latent input vector to produce an output from.

        Returns
        -------
        np.array
            Output produced by generator.
        """
        if z is None:
            z = self.sample(n=len(y))
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(self.device)
        inpt = self.concatenate(z, y).float().to(self.device)
        if who == "GeneratorX_Y":
            sample = self.generatorX_Y(inpt)
        elif who == "GeneratorY_X":
            sample = self.generatorY_X(inpt)
        else:
            raise ValueError("`who` must be one of ['GeneratorX_Y', 'GeneratorY_X']. Given: {}.".format(who))
        if self.training:
            return sample
        return sample.detach().cpu().numpy()

    def predict(self, x, y, who="AdversaryX_Y"):
        """ Use the critic / discriminator to predict if input is real / fake.

        Parameters
        ----------
        x : np.array
            Images or samples to be predicted.
        y : np.array
            Labels for outputs to be predicted.

        Returns
        -------
        np.array
            Array with one output per x indicating the realness of an input.
        """
        inpt = self.concatenate(x, y).float().to(self.device)
        if who == "AdversaryX_Y":
            return self.adversaryX_Y(inpt)
        elif who == "AdversaryY_X":
            return self.adversaryY_X(inpt)
        else:
            raise ValueError("`who` must be one of ['AdversaryX_Y', 'AdversaryY_X']. Given: {}.".format(who))

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        if who == "Autoencoder":
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "AdversaryX_Y":
            losses = self._calculate_adversaryX_Y_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        elif who == "AdversaryY_X":
            losses = self._calculate_adversaryY_X_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            losses.update(self._calculate_adversaryX_Y_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch))
            losses.update(self._calculate_adversaryY_X_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch))
        return losses

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_imagesX_Y = self.generate(z=Z_batch, y=X_batch, who="GeneratorX_Y")
        fake_imagesY_X = self.generate(z=Z_batch, y=y_batch, who="GeneratorY_X")

        reconstructedX_Y_X = self.generate(z=Z_batch, y=fake_imagesX_Y, who="GeneratorY_X")
        reconstructedY_X_Y = self.generate(z=Z_batch, y=fake_imagesY_X, who="GeneratorX_Y")

        fake_predictionsX_Y = self.predict(x=fake_imagesX_Y, y=X_batch, who="AdversaryX_Y")
        fake_predictionsY_X = self.predict(x=fake_imagesY_X, y=X_batch, who="AdversaryY_X")

        gen_loss_fakeX_Y = self.loss_functions["Adversary"](
            fake_predictionsX_Y, torch.ones_like(fake_predictionsX_Y, requires_grad=False)
        )
        gen_loss_fakeY_X = self.loss_functions["Adversary"](
            fake_predictionsY_X, torch.ones_like(fake_predictionsY_X, requires_grad=False)
        )
        gen_loss_reconstructionX_Y_X= self.loss_functions["Reconstruction"](
            reconstructedX_Y_X, X_batch
        )
        gen_loss_reconstructionY_X_Y= self.loss_functions["Reconstruction"](
            reconstructedY_X_Y, y_batch
        )

        gen_loss = gen_loss_fakeX_Y + gen_loss_fakeY_X + self.lambda_x*(gen_loss_reconstructionX_Y_X + gen_loss_reconstructionY_X_Y)
        return {
            "Autoencoder": gen_loss,
            "GeneratorX_Y_fake": gen_loss_fakeX_Y,
            "GeneratorY_X_fake": gen_loss_fakeY_X,
            "ReconstructionX_Y_X": self.lambda_x*gen_loss_reconstructionX_Y_X,
            "ReconstructionY_X_Y": self.lambda_x*gen_loss_reconstructionY_X_Y,
        }

    def _calculate_adversaryX_Y_loss(self, X_batch, Z_batch, y_batch):
        fake_imagesX_Y = self.generate(z=Z_batch, y=X_batch, who="GeneratorX_Y").detach()
        fake_predictionsX_Y = self.predict(x=fake_imagesX_Y, y=X_batch, who="AdversaryX_Y")
        real_predictionsX_Y = self.predict(x=y_batch, y=X_batch, who="AdversaryX_Y")

        adv_loss_fake = self.loss_functions["Adversary"](
            fake_predictionsX_Y, torch.zeros_like(fake_predictionsX_Y, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictionsX_Y, torch.ones_like(real_predictionsX_Y, requires_grad=False)
        )

        adv_loss = 1/2*(adv_loss_fake + adv_loss_real)
        return {
            "AdversaryX_Y": adv_loss,
            "AdversaryX_Y_fake": adv_loss_fake,
            "AdversaryX_Y_real": adv_loss_real,
            "RealFakeRatioX_Y": adv_loss_real / adv_loss_fake
        }

    def _calculate_adversaryY_X_loss(self, X_batch, Z_batch, y_batch):
        fake_imagesY_X = self.generate(z=Z_batch, y=y_batch, who="GeneratorY_X").detach()
        fake_predictionsY_X = self.predict(x=fake_imagesY_X, y=y_batch, who="AdversaryY_X")
        real_predictionsY_X = self.predict(x=X_batch, y=y_batch, who="AdversaryY_X")

        adv_loss_fake = self.loss_functions["Adversary"](
            fake_predictionsY_X, torch.zeros_like(fake_predictionsY_X, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictionsY_X, torch.ones_like(real_predictionsY_X, requires_grad=False)
        )

        adv_loss = 1/2*(adv_loss_fake + adv_loss_real)
        return {
            "AdversaryY_X": adv_loss,
            "AdversaryY_X_fake": adv_loss_fake,
            "AdversaryY_X_real": adv_loss_real,
            "RealFakeRatioY_X": adv_loss_real / adv_loss_fake
        }

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversary":
                if self.adv_type == "Critic":
                    for p in self.adversary.parameters():
                        p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]