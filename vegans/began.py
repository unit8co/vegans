import torch

from .gan import GAN


class BEGAN(GAN):
    """
    BEGAN: balancing generator and discriminator

    https://arxiv.org/abs/1703.10717
    """

    def __init__(self, generator,
                 discriminator,
                 dataloader,
                 optimizer_D=None,
                 optimizer_G=None,
                 nz=100,
                 device='cpu',
                 ngpu=0,
                 fixed_noise_size=64,
                 nr_epochs=5,
                 save_every=500,
                 print_every=50,
                 init_weights=False,
                 lr_decay_every=None):
        super().__init__(generator, discriminator, dataloader, optimizer_D, optimizer_G, nz, device, ngpu,
                         fixed_noise_size, nr_epochs, save_every, print_every, init_weights)
        self.lr_decay_every = lr_decay_every

    def _adjust_learning_rate(self, optimizer, niter):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (0.95 ** (niter // self.lr_decay_every))

        return optimizer

    def train(self, gamma=0.75, lambda_k=0.001, k=0.0):
        """

        :param gamma:
        :param lambda_k:
        :param k:
        :return:
        """
        for epoch in range(self.nr_epochs):
            for minibatch_iter, (data, _) in enumerate(self.dataloader):

                real = data.to(self.device)
                batch_size = real.size(0)

                """ Train the generator
                """
                self.optimizer_G.zero_grad()

                # generate fake
                noise = torch.randn(batch_size, self.nz, device=self.device)
                fake = self.generator(noise).detach()

                loss_G = torch.mean(torch.abs(self.discriminator(fake) - fake))

                loss_G.backward()
                self.optimizer_G.step()

                """ Train the discriminator
                """
                self.optimizer_D.zero_grad()

                real_D = self.discriminator(real)
                fake_D = self.discriminator(fake.detach())

                loss_D_real = torch.mean(torch.abs(real_D - real))
                loss_D_fake = torch.mean(torch.abs(fake_D - fake.detach()))
                loss_D = loss_D_real - k * loss_D_fake

                loss_D.backward()
                self.optimizer_D.step()

                """ Update weights
                """
                diff = gamma * loss_D_real.item() - loss_D_fake.item()

                # Update weight term
                k = k + lambda_k * diff
                k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

                # Update convergence metric
                m = loss_D_real.item() + abs(diff)

                # Learning rate decay, optional
                if self.lr_decay_every is not None:
                    self.optimizer_D = self._adjust_learning_rate(self.optimizer_D, minibatch_iter)
                    self.optimizer_G = self._adjust_learning_rate(self.optimizer_G, minibatch_iter)

                # Finish iteration
                self._end_iteration(epoch, minibatch_iter, loss_G.item(), loss_D.item(), M=m, K=k)

        return self.samples, self.D_losses, self.G_losses
