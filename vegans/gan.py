from abc import ABC, abstractmethod
import torch
import torch.optim as optim
import torch.nn as nn
import time


class GAN(ABC):
    """
    The base class of all GANs
    TODO: checkpointing
    """
    def __init__(self,
                 generator,
                 discriminator,
                 dataloader,
                 optimizer_D=None,
                 optimizer_G=None,
                 nz=100,
                 ngpu=1,
                 fixed_noise_size=64,
                 nr_epochs=5,
                 save_every=500,
                 print_every=50,
                 init_weights=False):
        """
        :param generator: G
        :param discriminator: D
        :param dataloader: A [torch.utils.data.DataLoader] containing training data
        :param optimizer_D: A [torch.optim.Optimizer] for D
        :param optimizer_G: A [torch.optim.Optimizer] for G
        :param nz: the size of the latent space
        :param ngpu: the number of GPUs to use
        :param fixed_noise_size: the number of samples to save with fixed noise
        :param nr_epochs: the number of epochs with which to train
        :param save_every: save some samples every [save_every] iterations
        :param print_every: prints current metrics every [print_every] iterations
        :param init_weights: whether to re-initialize the weights of G and D when building this GAN
        """

        self.generator = generator
        self.discriminator = discriminator
        self.nz = nz
        self.dataloader = dataloader
        self.nr_epochs = nr_epochs
        self.save_every = save_every
        self.print_every = print_every

        # TODO: several
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        print('device: {}'.format(self.device))

        self.optimizer_D = optimizer_D if optimizer_D is not None else self._default_optimizers()[0]
        self.optimizer_G = optimizer_G if optimizer_G is not None else self._default_optimizers()[1]

        # Optionally (re-)init G and D
        if init_weights:
            self._default_weights_init()

        # Create batch of latent vectors that we will use to generate samples
        self.fixed_noise = torch.randn(fixed_noise_size, self.nz, device=self.device)

        # Keep track of losses and samples over the course of training:
        self.D_losses, self.G_losses, self.samples = dict(), dict(), dict()

        # state of iterations
        self.nr_iters_since_last_print = 0
        self.last_print_time = None
        self.global_iter = 0
        self.last_D_loss = None
        self.last_G_loss = None

    def _init_structs(self,):
        self.D_losses, self.G_losses, self.samples = dict(), dict(), dict()

    def get_training_results(self,):
        """
        :return: A tuple of 3 dictionaries: (samples, G_losses, D_losses). Each dictionary is keyed
                 by a (epoch, minibatch_iter) tuple.
                 - [samples] contains the samples produced from [fixed_noise]
                 - [G_losses] contains the generator losses
                 - [D_losses] contains the discriminator/critic losses
                 This is meant to be called after (some amount of) training
        """
        return self.samples, self.D_losses, self.G_losses

    def _default_weights_init(self,):
        """
        Based on DCGAN
        """
        for m in [self.generator, self.discriminator]:
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def _default_optimizers(self,):
        """
        Implementations can provide different default optimizers here.
        Those come from DCGAN
        """
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer_D, optimizer_G

    def _end_iteration(self, epoch, minibatch_iter, G_loss=None, D_loss=None, D_x=None, D_G_z1=None, D_G_z2=None):
        """
        Some boilerplate work done at each iteration (printing, saving, timing)

        Not all metrics (losses, accuracies) are present for both networks at each iteration
        """

        # increment global iteration counter:
        self.global_iter += 1

        # save losses, and possibly some samples obtained from fixed noise:
        if D_loss is not None:
            self.D_losses[(epoch, minibatch_iter)] = D_loss
            self.last_D_loss = D_loss
        if G_loss is not None:
            self.G_losses[(epoch, minibatch_iter)] = G_loss
            self.last_G_loss = G_loss
        if (self.global_iter % self.save_every == 0) or \
                ((epoch == self.nr_epochs - 1) and (minibatch_iter == len(self.dataloader) - 1)):
            with torch.no_grad():
                self.samples[(epoch, minibatch_iter)] = self.generator(self.fixed_noise).detach().cpu()

        # print every [self.print_every] iteration
        if self.nr_iters_since_last_print == self.print_every:
            def _format_none(v):
                return '%.3f' % v if v is not None else '-'

            now = time.time()
            if self.last_print_time is not None:
                avg_iter_per_s = self.nr_iters_since_last_print / (now - self.last_print_time)
            else:
                avg_iter_per_s = None
            self.last_print_time = now
            self.nr_iters_since_last_print = 0

            s_accum = '[%d/%d][%d/%d](%s iter/s)' % (epoch, self.nr_epochs, minibatch_iter,
                                                     len(self.dataloader), _format_none(avg_iter_per_s))
            for n, v in [('Loss_D', self.last_D_loss), ('Loss_G', self.last_G_loss), ('D(x)', D_x),
                         ('D(G(z1))', D_G_z1), ('D(G(z2))', D_G_z2)]:
                if v is not None:
                    s_accum += '\t%s: %.3f' % (n, v)
            print(s_accum)
        self.nr_iters_since_last_print += 1

    @abstractmethod
    def train(self,):
        """
        :return:
        """
        pass
