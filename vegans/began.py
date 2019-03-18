import torch

from .gan import GAN


class BEGAN(GAN):
    """
    BEGAN: balancing generator and discriminator

    https://arxiv.org/abs/1703.10717
    """

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
                # M = (loss_D_real + torch.abs(diff)).data

                # Finish iteration
                self._end_iteration(epoch, minibatch_iter, loss_G.item() if loss_G is not None else None, loss_D.item())

        return self.samples, self.D_losses, self.G_losses
