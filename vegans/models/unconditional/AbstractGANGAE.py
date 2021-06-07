import torch

from torch.nn import BCELoss
from vegans.utils import WassersteinLoss
from vegans.utils.networks import Generator, Adversary, Encoder
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel


class AbstractGANGAE(AbstractGenerativeModel):
    """ Abstract class for GAN with structure of one generator, one discriminator / critic and
    one encoder. Examples are the `LRGAN`, `VAEGAN` and `BicycleGAN`.

    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
    encoder : nn.Module
        Encoder architecture. Produces predictions in the latent space.
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
    feature_layer : torch.nn.*
        Output layer used to compute the feature loss. Should be from either the discriminator or critic.
        If `feature_layer` is not None, the original generator loss is replaced by a feature loss, introduced
        [here](https://arxiv.org/abs/1606.03498v1).
    fixed_noise_size : int
        Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
        subdirectory, the tensorboard images tab and the samples in get_training_results().
    lambda_grad: float
        Weight for the reconstruction loss of the gradients. Pushes the norm of the gradients to 1.
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
            encoder,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder=None,
            ngpu=0,
            secure=True,
            _called_from_conditional=False):

        self.adv_type = adv_type
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversary = Adversary(adversary, input_size=x_dim, adv_type=adv_type, device=device, ngpu=ngpu, secure=secure)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu, secure=secure)
        self.neural_nets = {
            "Generator": self.generator, "Adversary": self.adversary, "Encoder": self.encoder
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, ngpu=ngpu, folder=folder, secure=secure
        )
        self.hyperparameters["adv_type"] = adv_type
        if not _called_from_conditional and self.secure:
            assert self.generator.output_size == self.x_dim, (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            loss_functions = {"Generator": BCELoss(), "Adversary": BCELoss()}
        elif self.adv_type == "Critic":
            loss_functions = {"Generator": WassersteinLoss(), "Adversary": WassersteinLoss()}
        else:
            raise NotImplementedError("'adv_type' must be one of Discriminator or Critic.")
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def calculate_losses(self, X_batch, Z_batch, who=None):
        """ Calculates the losses for GANs using a 1v1 architecture.

        This method is called within the `AbstractGenerativeModel` main `fit()` loop.

        Parameters
        ----------
        X_batch : torch.Tensor
            Current x batch.
        Z_batch : torch.Tensor
            Current z batch.
        who : None, optional
            Name of the network that should be trained.
        """
        if who == "Generator":
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Adversary":
            losses = self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Encoder":
            losses = self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
            losses.update(self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch))
            losses.update(self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch))
        return losses

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversary":
                if self.adv_type == "Critic":
                    for p in self.adversary.parameters():
                        p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]


    #########################################################################
    # Utility functions
    #########################################################################
    def encode(self, x):
        return self.encoder(x)