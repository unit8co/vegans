import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils  # TODO: dynamically handle image types


class GAN:
    def __init__(self, generator, discriminator, nz=100):
        self.generator = generator
        self.discriminator = discriminator
        self.nz = nz

    def _weights_init(self,):
        # custom weights initialization called on generator and discriminator
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
              optimizerD=None,
              optimizerG=None,
              num_epochs=5,
              fixed_latent_batch_size=64,
              init_weights=True):
        """
        TODO:
            - checkpointing
            - flat noise vector
            - see if: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
              can help simplify implementation

        :param dataloader:
        :param ngpu:
        :param optimizerD:
        :param optimizerG:
        :param num_epochs:
        :param fixed_latent_batch_size:
        :return:
        """

        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        if init_weights:
            self._weights_init()

        # Initialize BCELoss function
        criterion = nn.BCELoss()


        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator

        # TODO: remove the last two dimensions here and in [noise].
        # TODO: this leaves responsibility of redimensioning to the generator
        fixed_noise = torch.randn(fixed_latent_batch_size, self.nz, 1, 1, device=device)
        # fixed_noise = torch.randn(fixed_latent_batch_size, self.nz, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        """ Default optimizers for G and D
            TODO: abstract function?
        """
        if optimizerD is None:
            optimizerD = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        if optimizerG is None:
            optimizerG = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


        """ Training Loop
        """
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting training Loop...")
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):

                """ (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    The BCE loss is: -[y log(x) + (1 - y) log(1 - x)], so we want to minimize it
                """

                ## Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

        return img_list, G_losses, D_losses
