# VeGANs

A library to easily train various existing GANs (Generative Adversarial Networks) in PyTorch.

This library targets mainly GAN users, who want to use existing GAN training techniques with their own generators/discriminators.
However researchers may also find the GAN base class useful for quicker implementation of new GAN training techniques.

The focus is on simplicity and providing reasonable defaults.

## How to install
You need python 3.5 or above. Then:
`pip install vegans`

## How to use
The basic idea is that the user provides discriminator and generator networks, and the library takes care of training them in a selected GAN setting:
```
from vegans import WGAN
from vegans.utils import plot_losses, plot_image_samples

netD = ### Your discriminator/critic (torch.nn.Module)
netG = ### Your generator (torch.nn.Module)
dataloader = ### Your dataloader (torch.utils.data.DataLoader)

# Build a Wasserstein GAN
gan = WGAN(netG, netD, dataloader, nr_epochs=20)

# train it
gan.train()

# vizualise results
img_list, D_losses, G_losses = gan.get_training_results()
plot_losses(G_losses, D_losses)
plot_image_samples(img_list, 50)
```

You can currently use the following GANs:
* `MMGAN`: [Classic minimax GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), in its non-saturated version
* `WGAN`: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
* `WGANGP`: [Wasserstein GAN with gradient penalty](https://arxiv.org/abs/1704.00028)

### Slightly More Details:
All of these GAN objects inherit from a `GAN` base class. When building any such GAN, you must give in argument a generator and discriminator networks (some `torch.nn.Module`), as well as a `torch.utils.data.DataLoader`. In addition, you can specify some parameters supported by all GAN implementations:
* `optimizer_D` and `optimizer_G`: some PyTorch optimizers (from `torch.optim`) for the discriminator and generator networks. By defaults those are set with default optimization parameters suggested in the original papers.
* `nr_epochs`: the number of epochs (default: 5)
* `nz`: size of the noise vector (input of the generator) - by default `nz=100`.
* `save_every`: VeGANs will store some samples produced by the generator every `save_every` iteration. Default: 500
* `fixed_noise_size`:  The number of samples to save (from fixed noise vectors)
* `print_every`: The number of iterations between printing training progress. Default: 50

Finally, when calling `train()` you can specify some parameters specific to each GAN. For example, for the Wasserstein GAN we can do:
```
gan = WGAN(netG, netD, dataloader)
gan.train(clip_value=0.1)
```
This will train a Wasserstein GAN with clipping values of `0.1` (instead of the default `0.01`).

If you are researching new GAN training algorithms, you may find it useful to inherit from the `GAN` base class.

### Learn more:
Currently the best way to learn more about how to use VeGANs is to have a look at the example [notebooks](https://github.com/unit8co/vegans/tree/master/notebooks).
You can start with this [simple example](https://github.com/unit8co/vegans/blob/master/notebooks/00_univariate_gaussian.ipynb) showing how to sample from a univariate Gaussian using a GAN.

## Contribute
PRs and suggestions are welcome.

## Credits
Some of the code has been inspired by some existing GAN implementations:
* https://github.com/eriklindernoren/PyTorch-GAN
* https://github.com/martinarjovsky/WassersteinGAN
* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
