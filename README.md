# VeGANs (might be merged with the current pip package vegan in the near future)

A library to easily train various existing GANs (Generative Adversarial Networks) in PyTorch.

This library targets mainly GAN users, who want to use existing GAN training techniques with their own generators/discriminators.
However researchers may also find the GAN base class useful for quicker implementation of new GAN training techniques.

The focus is on simplicity and providing reasonable defaults.

## How to install
You need python 3.5 or above. Then:
~~`pip install vegans`~~ (Not yet)

## How to use
The basic idea is that the user provides discriminator and generator networks, and the library takes care of training them in a selected GAN setting:
```
from vegans.GAN import WassersteinGAN
from vegans.utils import plot_losses, plot_images

generator = ### Your generator (torch.nn.Module)
adversariat = ### Your discriminator/critic (torch.nn.Module)
X_train = ### Your dataloader (torch.utils.data.DataLoader) or pd.DataFrame

z_dim = 64
x_dim = X_train.shape[1:] # [nr_channels, height, width]

# Build a WassersteinGAN
gan = WassersteinGAN(generator, discriminator, z_dim, x_dim)
gan.summary() # optional

# Fit the WassersteinGAN
gan.fit(X_train)

# Vizualise results
images, losses = gan.get_training_results()
images = images.reshape(-1, *samples.shape[2:]) # remove nr_channels
plot_images(images)
plot_losses(losses)
```

You can currently use the following GANs:
* `VanillaGAN`: [Classic minimax GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), in its non-saturated version
* `WassersteinGAN`: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
* `WassersteinGANGP`: [Wasserstein GAN with gradient penalty](https://arxiv.org/abs/1704.00028)
* `LSGAN`: [Least-Squares GAN](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf)
* `LRGAN`: [Latent-Regressor GAN](https://arxiv.org/pdf/1711.11586.pdf)

All current GAN implementations come with a conditional variant to allow for the usage of training labels to produce specific outputs:

- `ConditionalVanillaGAN`
- `ConditionalWassersteinGAN`
- ...
- `ConditionalPix2Pix`

This can either be used to pass a one hot encoded vector to predict a specific label (generate a certain number in case of mnist: [example_conditional.py](https://github.com/tneuer/GAN-pytorch/blob/main/examples/example_conditional.py)  or [03_mnist-conditional.ipynb](https://github.com/tneuer/GAN-pytorch/blob/main/notebooks/03_mnist-conditional.ipynb)) or it can also be a full image (when for example trying to rotate an image: [example_image_to_image.py](https://github.com/tneuer/GAN-pytorch/blob/main/examples/example_image_to_image.py)  or [04_mnist-image-to-image.ipynb](https://github.com/tneuer/GAN-pytorch/blob/main/notebooks/04_mnist-image-to-image.ipynb)).

Models can either be passed as `torch.nn.Sequential` objects or by defining custom architectures, see [example_input_formats.py](https://github.com/tneuer/GAN-pytorch/blob/main/examples/example_input_formats.py).

Also look at the [jupyter notebooks](https://github.com/tneuer/GAN-pytorch/tree/main/notebooks) for better visualized examples and how to use the library.



### Slightly More Details:

All of the GAN objects inherit from a `AbstractGenerativeModel` base class. When building any such GAN, you must pass generator as well as discriminator networks (some `torch.nn.Module`), as well as a the dimensions of the latent space `z_dim` and input dimension of the images `x_dim`. In addition, you can specify some parameters supported by all GAN implementations:
* `optim`: The optimizer to use for all networks during training. If `None` a default optimizer (probably either `torch.optim.Adam` or `torch.optim.RMSprop`) is chosen by the specific model. A `dict` type with appropriate keys can be passed to specify different optimizers for different networks.
* `optim_kwargs`:  The optimizer default arguments. A `dict` type with appropriate keys can be passed to specify different optimizer keyword arguments for different networks.
* `fixed_noise_size`: The number of samples to save (from fixed noise vectors). These are saved within Tensorboard (if `enable_tensorboard=True` during fitting) and in the `Model/images` subfolder.
* `device`: "cuda" (GPU) or "cpu" depending on the available resources.
* `folder`: Folder which will contain all results of the network (architecture, model.torch, images, loss plots, etc.). An existing folder will never be deleted or overwritten. If the folder already exists a new folder will be created with the given name + current time stamp.
* `ngpu`: Number of gpus used during training

The fit function takes the following optional arguments:

- `epochs`: Number of epochs to train the algorithm. Default: 5
- `batch_size`: Size of one batch. Should not be too large: Default: 32
- `steps`: How often one network should be trained against another. Must be `dict` type with appropriate names.
- `print_every`: Determines after how many batches a message should be printed to the console informing about the current state of training. String indicating fraction or multiples of epoch can be given. I.e. "0.25e" = four times per epoch, "2e" after two epochs. Default: 100
- `save_model_every`: Determines after how many batches the model should be saved. String indicating fraction or multiples of epoch can be given. I.e. "0.25e" = four times per epoch, "2e" after two epochs. Models will be saved in subdirectory `save.folder`+"/models". Default: None
- `save_images_every`: Determines after how many batches sample images and loss curves should be saved. String indicating fraction or multiples of epoch can be given. I.e. "0.25e" = four times per epoch, "2e" after two epochs. Images will be saved in subdirectory `save.folder`+"/images".  Default: None
- `save_losses_every`: Determines after how many batches the losses should be calculated and saved. Figure is shown after `save_images_every` . String indicating fraction or multiples of epoch can be given. I.e. "0.25e" = four times per epoch, "2e" after two epochs. Default: "1e"
- `enable_tensorboard`: Determines after how many batches a message should be printed to the console informing about the current state of training. Tensorboard information will be saved in subdirectory `save.folder`+"/tensorboard".  Default: True



If you are researching new GAN training algorithms, you may find it useful to inherit from the `AbstractGenerativeModel` or  `AbstractConditionalGenerativeModel` base class.

### Learn more:

Currently the best way to learn more about how to use VeGANs is to have a look at the example [notebooks](https://github.com/tneuer/GAN-pytorch/tree/main/notebooks).
You can start with this [simple example](https://github.com/tneuer/GAN-pytorch/blob/main/notebooks/00_univariate_gaussian.ipynb) showing how to sample from a univariate Gaussian using a GAN.
Alternatively, can run example [scripts](https://github.com/tneuer/GAN-pytorch/tree/main/examples).

## Contribute
PRs and suggestions are welcome. Look [here](https://github.com/unit8co/vegans/blob/master/CONTRIBUTING) for more details on the setup.

## Credits
Some of the code has been inspired by some existing GAN implementations:
* https://github.com/eriklindernoren/PyTorch-GAN
* https://github.com/martinarjovsky/WassersteinGAN
* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## TODO

- GAN Implementations (sorted by priority)
  - VAEGAN
  - BicycleGAN
  - InfoGAN
  - BEGAN
  - EBGAN
  - CycleGAN
  - WassersteinGAN SpectralNorm
  - DiscoGAN
  - CycleGAN
  - Adversarial Autoencoder
- Layers 
  - Inception
  - Residual Block
  - Minibatch discrimination
- Other

  - Write tests
  
  - Feature loss
  
  - enable Wasserstein loss for all architectures (when it makes sense)
  
  - Do not save Discriminator
  
    



- Done
  - ~~Test dependencies~~
  - ~~LR-GAN~~
  - ~~Least Squares GAN~~
  - ~~Include sources in jupyter~~
  - ~~Make all examples work nicely~~
  - ~~Implement Pix2Pix architecture: https://blog.eduonix.com/artificial-intelligence/pix2pix-gan/~~
  - ~~Include images in jupyter~~
  - ~~Pix2Pix~~
  - ~~Check output dim (generator, encoder)~~
  - ~~Improve Doc for networks~~
  - ~~Rename AbstractGAN1v1 -> AbstractAbstractGAN1v1~~
  - ~~Rename AbstractConditionalGAN1v1 -> AbstractAbstractConditionalGAN1v1~~
  - ~~Rename AbstractGenerativeModel -> AbstractAbstractGenerativeModel~~
  - ~~Rename AbstractConditionalGenerativeModel -> AbstractAbstractConditionalGenerativeModel~~
  - ~~return numpy array instead o tensor for generate.~~
  - ~~Automatically use evaluation mode~~











