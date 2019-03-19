import torch
import torch.nn as nn
from .gan import GAN


class MMGAN(GAN):
    """
    Minimax GAN (classic GAN), in its non-saturated version (i.e., generator loss is log(D(x)) ).
    Also called NS-GAN sometimes.

    https://arxiv.org/abs/1406.2661
    """

    def train(self,):

        criterion = nn.BCELoss()

        """ Training Loop
        """
        for epoch in range(self.nr_epochs):
            for minibatch_iter, (data, _) in enumerate(self.dataloader):

                real = data.to(self.device)
                batch_size = real.size(0)
                real_labels = torch.full((batch_size,), 1.0, device=self.device)
                fake_labels = torch.full((batch_size,), 0.0, device=self.device)

                """ (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    The BCE loss is: -[y log(x) + (1 - y) log(1 - x)], so we want to minimize it
                """
                # Train with real
                self.optimizer_D.zero_grad()
                output = self.discriminator(real).view(-1)
                errD_real = criterion(output, real_labels)  # loss on real batch
                errD_real.backward()  # gradients for real batch
                D_x = output.mean().item()

                # Train with fake
                noise = torch.randn(batch_size, self.nz, device=self.device)
                fake = self.generator(noise)
                output = self.discriminator(fake.detach()).view(-1)
                errD_fake = criterion(output, fake_labels)  # loss on fake batch
                errD_fake.backward()  # gradients for fake batch
                D_G_z1 = output.mean().item()
                D_loss = errD_real + errD_fake  # total loss (sum of real and fake losses)
                self.optimizer_D.step()  # Update D

                """ (2) Update G network: maximize log(D(G(z)))
                """
                self.optimizer_G.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                G_loss = criterion(output, real_labels)  # loss. Fake labels are real for generator cost
                G_loss.backward()
                D_G_z2 = output.mean().item()
                self.optimizer_G.step()  # Update G

                # Finish iteration
                self._end_iteration(epoch, minibatch_iter, G_loss.item(), D_loss.item(), D_x, D_G_z1, D_G_z2)

        return self.samples, self.D_losses, self.G_losses
