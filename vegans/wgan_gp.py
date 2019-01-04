import torch
from .gan import GAN


class WGANGP(GAN):
    """
    Wasserstein GAN with gradient penalty
    https://arxiv.org/abs/1704.00028
    """

    def train(self, critic_iters=5, long_critic_iters=100, lambda_gp=10):
        """
        :param critic_iters:
        :param long_critic_iters:
        :param clip_value:
        :return:
        """

        def _grad_penalty(real, fake):
            """
            Computes the gradient penalty.
            The gradient is taken for linear interpolations between real and fake samples.
            """
            assert real.size() == fake.size(), 'real and fake mini batches must have same size'
            batch_size = real.size(0)
            epsilon = torch.rand(batch_size, *[1 for _ in range(real.dim()-1)], device=self.device)
            x_hat = (epsilon * real + (1. - epsilon) * fake).requires_grad_(True)
            output = self.discriminator(x_hat)
            grads = torch.autograd.grad(
                outputs=output,
                inputs=x_hat,
                grad_outputs=torch.ones((batch_size,), device=self.device)
            )[0]
            return ((grads.norm(2, dim=1) - 1) ** 2).mean()

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
                loss_D_with_penalty = loss_D + lambda_gp * _grad_penalty(real, fake)

                loss_D_with_penalty.backward()
                self.optimizer_D.step()

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
