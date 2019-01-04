import torch
import torch.optim as optim
from .gan import GAN


class WGANGP(GAN):
    """
    Original Wasserstein GAN
    https://arxiv.org/abs/1701.07875
    """

    def _default_optimizers(self,):
        """
        The WGAN paper proposes RMSprop
        """
        optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=0.00005)
        optimizer_G = optim.RMSprop(self.generator.parameters(), lr=0.00005)
        return optimizer_D, optimizer_G

    def train(self, critic_iters=5, long_critic_iters=100, clip_value=0.01):
        """

        :param critic_iters:
        :param long_critic_iters:
        :param clip_value:
        :return:
        """

        gen_iters = 0  # the generator is not trained every iteration
        for epoch in range(self.nr_epochs):
            for minibatch_iter, (data, _) in enumerate(self.dataloader):

                # the number of mini batches we'll train the critic before training the generator
                if gen_iters < 25 or gen_iters % 500 == 0:
                    D_iters = long_critic_iters
                else:
                    D_iters = critic_iters

                real = data.to(self.device)
                batch_size = real.size(0)

                """ Train the critic
                """
                self.optimizer_D.zero_grad()
                noise = torch.randn(batch_size, self.nz, device=self.device)
                fake = self.generator(noise).detach()

                # Sign is inverse of paper because in paper it's a maximization problem
                loss_D = self.discriminator(fake).mean() - self.discriminator(real).mean()

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                loss_G = None
                if self.global_iter % D_iters == 0:
                    """ Train the generator every [Diters]
                    """
                    self.optimizer_G.zero_grad()
                    fake = self.generator(noise)
                    loss_G = -torch.mean(self.discriminator(fake))
                    loss_G.backward()
                    self.optimizer_G.step()
                    gen_iters += 1

                # End iteration
                self._end_iteration(epoch, minibatch_iter, loss_G.item() if loss_G is not None else None, loss_D.item())

        return self.samples, self.D_losses, self.G_losses
