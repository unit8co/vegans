import torch
import torch.nn as nn
import torch.optim as optim


class WGAN:
    def __init__(self, generator, discriminator, nz=100):
        self.generator = generator
        self.discriminator = discriminator
        self.nz = nz

    def train(self,
              dataloader,
              ngpu=0,
              optimizer_D=None,
              optimizer_G=None,
              num_epochs=5,
              fixed_noise_size=64,
              print_every=50,
              save_every=500,
              critic_iters=5,
              clip_value=0.01):
        """
        TODO: checkpointing

        :param dataloader:
        :param ngpu:
        :param optimizer_D:
        :param optimizer_G:
        :param num_epochs:
        :param fixed_latent_batch_size:
        :param init_weights:
        :param print_every: print every [print_every] mini batches within an epoch
        :param save_every: save generated samples every [save_every] iterations. In addition, it also saves
                           samples generated during the last mini batch.
        :return:
        """

        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Create batch of latent vectors that we will use to generate samples
        fixed_noise = torch.randn(fixed_noise_size, self.nz, device=device)

        """ Default optimizers for G and D
            TODO: abstract function?
        """
        if optimizer_D is None:
            optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00005)
        if optimizer_G is None:
            optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=0.00005)

        """ Training Loop
        """
        # Structures keep track of progress. <(epoch, mini_batch), value>
        samples_list = dict()
        G_losses = dict()
        D_losses = dict()

        iters = 0
        gen_iters = 0

        print("Starting training Loop...")
        for epoch in range(num_epochs):
            for i, (data, _) in enumerate(dataloader):

                # the number of mini batches we'll train the critic before training the generator
                if gen_iters < 25 or gen_iters % 500 == 0:
                    D_iters = 100
                else:
                    D_iters = critic_iters

                real = data.to(device)
                batch_size = real.size(0)

                """ Train the critic
                """
                optimizer_D.zero_grad()
                noise = torch.randn(batch_size, self.nz, device=device)
                fake = self.generator(noise).detach()

                # Note: sign is inverse of paper because we minimize the loss
                # (instead of maximizing as in paper)
                loss_D = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
                D_losses[(epoch, i)] = loss_D.item()

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                if iters % D_iters == 0:
                    """ Train the generator every [Diters]
                    """
                    optimizer_G.zero_grad()
                    fake = self.generator(noise)
                    loss_G = -torch.mean(self.discriminator(fake))
                    G_losses[(epoch, i)] = loss_G.item()
                    loss_G.backward()
                    optimizer_G.step()
                    gen_iters += 1

                # Output training stats
                if i % print_every == 0:
                    last_G_loss = G_losses[max(G_losses.keys())] if len(G_losses) > 0 else float('nan')
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch, num_epochs, i, len(dataloader),
                          loss_D.item(), last_G_loss))

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % save_every == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    # samples_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    samples_list[(epoch, i)] = fake

                iters += 1

        return samples_list, G_losses, D_losses
