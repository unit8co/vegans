from abc import ABC, abstractmethod
import torch
import torch.optim as optim


class GAN(ABC):
    """
    The base class of all GANs
    """
    def __init__(self, generator, discriminator, nz=100, ngpu=1, optimizer_D=None, optimizer_G=None):
        self.generator = generator
        self.discriminator = discriminator
        self.nz = nz

        # TODO: several
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.optimizer_D = optimizer_D if optimizer_D is not None else self._default_optimizers()[0]
        self.optimizer_G = optimizer_G if optimizer_G is not None else self._default_optimizers()[1]

    def _default_optimizers(self, ):
        """
        It is recommended (but not mandatory) for implementations to override these defaults
        """
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer_D, optimizer_G

    @abstractmethod
    def train(self,
              dataloader,
              num_epochs,
              save_every,
              fixed_noise_size,
              print_every):
        """
        :param dataloader: A [torch.utils.data.DataLoader] containing training data
        :param num_epochs: the number of epochs
        :param save_every: save some samples every [save_every] iterations
        :param fixed_noise_size: the number of samples to generate
        :param print_every: prints current metrics every [print_every] iterations
        :return: a tuple of 3 dictionaries: (samples, G_losses, D_losses). Each dictionary is keyed
                 by a (epoch, minibatch_iter) tuple.
                 - [samples] contains the samples produced from [fixed_noise]
                 - [G_losses] contains the generator losses
                 - [D_losses] contains the discriminator/critic losses
        """
        pass
