# VeGANs

A library providing various existing GANs in PyTorch.

This library targets mainly GAN users, who want to use existing GAN training techniques with their own generators/discriminators.
However researchers may also find the GAN base class useful for quicker implementation of new GAN training techniques.

The focus is on simplicity and providing reasonable defaults.

## How to install
`pip install vegans`

## How to use
The basic idea is that the user provides discriminator and generator networks, and the library takes care of training them in a selected GAN setting:
```
from vegans import WGAN
from vegans.utils import plot_losses, plot_image_samples

# Create your critic and generator
netD = Discriminator().to(device)
netG = Generator().to(device)

# Build a Wasserstein GAN
gan = WGAN(netG, netD, dataloader, ngpu=1, nr_epochs=20)

# train it
gan.train()

# vizualise results
img_list, D_losses, G_losses = gan.get_training_results()
plot_losses(G_losses, D_losses)
plot_image_samples(img_list, 50)
```

Currently the best way to learn more about how to use VeGANs is to have a look at the example [notebooks](https://github.com/unit8co/vegans).

## Contribute
PRs and suggestions are welcome.

## Credits
Some of the code has been inspired by some existing GAN implementations:
* https://github.com/eriklindernoren/PyTorch-GAN
* https://github.com/martinarjovsky/WassersteinGAN
* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
