import torch
import torch.nn as nn
import torch.optim as optim


class MMGAN:
    def __init__(self, generator, discriminator, nz=100):
        self.generator = generator
        self.discriminator = discriminator
        self.nz = nz

    def _weights_init(self,):
        # custom weights initialization called on generator and discriminator; based on DCGAN
        for m in [self.generator, self.discriminator]:
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def train(self,
              dataloader,
              ngpu=0,
              optimizer_D=None,
              optimizer_G=None,
              num_epochs=5,
              fixed_noise_size=64,
              init_weights=True,
              print_every=50,
              save_every=500):
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

        if init_weights:
            self._weights_init()

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to generate samples
        fixed_noise = torch.randn(fixed_noise_size, self.nz, device=device)

        """ Default optimizers for G and D
            TODO: abstract function?
        """
        if optimizer_D is None:
            optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        if optimizer_G is None:
            optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        """ Training Loop
        """
        # Structures keep track of progress. <(epoch, mini_batch), value>
        samples_list = dict()
        G_losses = dict()
        D_losses = dict()
        iters = 0

        print("Starting training Loop...")
        for epoch in range(num_epochs):
            for i, (data, _) in enumerate(dataloader):

                real = data.to(device)
                batch_size = real.size(0)
                real_labels = torch.full((batch_size,), 1.0, device=device)
                fake_labels = torch.full((batch_size,), 0.0, device=device)

                """ (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    The BCE loss is: -[y log(x) + (1 - y) log(1 - x)], so we want to minimize it
                """
                ## Train with real
                optimizer_D.zero_grad()
                output = self.discriminator(real)
                errD_real = criterion(output, real_labels)  # loss on real batch
                errD_real.backward()  # gradients for real batch
                D_x = output.mean().item()

                ## Train with fake
                noise = torch.randn(batch_size, self.nz, device=device)
                fake = self.generator(noise)
                output = self.discriminator(fake.detach())
                errD_fake = criterion(output, fake_labels)  # loss on fake batch
                errD_fake.backward()  # gradients for fake batch
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake  # total loss (sum of real and fake losses)
                optimizer_D.step()  # Update D

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                optimizer_G.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                errG = criterion(output, real_labels)  # loss. Fake labels are real for generator cost
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizer_G.step()  # Update G

                # Output training stats
                if i % print_every == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                G_losses[(epoch, i)] = errG.item()
                D_losses[(epoch, i)] = errD.item()

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % save_every == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    # samples_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    samples_list[(epoch, i)] = fake
                iters += 1

        return samples_list, G_losses, D_losses
