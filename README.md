# VeGANs (might be merged with the current pip package vegan in the near future)

A library to easily train various existing GANs (Generative Adversarial Networks) in PyTorch.

This library targets mainly GAN users, who want to use existing GAN training techniques with their own generators/discriminators.
However researchers may also find the GAN base class useful for quicker implementation of new GAN training techniques.

The focus is on simplicity and providing reasonable defaults.

## How to install
You need python 3.5 or above. Then:
~~`pip install vegans`~~ (Not yet)

## How to use
The basic idea is that the user provides discriminator / critic and generator networks (additionally an encoder if needed), and the library takes care of training them in a selected GAN setting. Several different use cases are shown below.

#### Unsupervised Learning

```python
from vegans.GAN import WassersteinGAN
import vegans.utils.utils as utils
import vegans.utils.loading as loading

datapath =  "./data/mnist/"
X_train, y_train, X_test, y_test = (
    loading.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
)
X_train = X_train.reshape((-1, 1, 32, 32)) # required shape
X_test = X_test.reshape((-1, 1, 32, 32))
x_dim = X_train.shape[1:] # [nr_channels, height, width]
z_dim = 64

# Define your own architectures here. You can use a Sequential model or an object
# inheriting from torch.nn.Module.
generator = loading.load_example_generator(x_dim=x_dim, z_dim=z_dim)
critic = loading.load_example_adversariat(x_dim=x_dim, z_dim=z_dim, adv_type="Critic")

gan = WassersteinGAN(
    generator=generator, adversariat=critic,
    z_dim=z_dim, x_dim=x_dim, folder=None
)
gan.summary() # optional, shows architecture
gan.fit(X_train, enable_tensorboard=False)

# Vizualise results
images, losses = gan.get_training_results()
images = images.reshape(-1, *images.shape[2:]) # remove nr_channels for plotting
utils.plot_images(images)
utils.plot_losses(losses)

# Sample new images, you can also pass a specific noise vector
samples = gan.generate(n=36)
samples = samples.reshape(-1, *samples.shape[2:]) # remove nr_channels for plotting
utils.plot_images(samples)
```

You can currently use the following GANs:
* `AAE`: [Adversarial Auto-Encoder](https://arxiv.org/pdf/1511.05644.pdf)

* `BicycleGAN`: [BicycleGAN](https://arxiv.org/pdf/1711.11586.pdf)

* `EBGAN`: [Energy-Based GAN](https://arxiv.org/pdf/1609.03126.pdf)

* `KLGAN`: Kullback-Leib GAN

* `LRGAN`: [Latent-Regressor GAN](https://arxiv.org/pdf/1711.11586.pdf)

* `LSGAN`: [Least-Squares GAN](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf)

* `VAEGAN`: [Variational Auto-Encoder GAN](https://arxiv.org/pdf/1512.09300.pdf)

* `VanillaGAN`: [Classic minimax GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), in its non-saturated version

* `VanillaVAE`: [Variational Auto-Encoder](https://arxiv.org/pdf/1512.09300.pdf)

* `WassersteinGAN`: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

* `WassersteinGANGP`: [Wasserstein GAN with gradient penalty](https://arxiv.org/abs/1704.00028)

  

All current GAN implementations come with a conditional variant to allow for the usage of training labels to produce specific outputs:

- `ConditionalVanillaGAN`
- `ConditionalWassersteinGAN`
- ...
- `ConditionalPix2Pix`

This can either be used to pass a one hot encoded vector to predict a specific label (generate a certain number in case of mnist: [example_conditional.py](https://github.com/tneuer/GAN-pytorch/blob/main/examples/example_conditional.py)  or [03_mnist-conditional.ipynb](https://github.com/tneuer/GAN-pytorch/blob/main/notebooks/03_mnist-conditional.ipynb)) or it can also be a full image (when for example trying to rotate an image: [example_image_to_image.py](https://github.com/tneuer/GAN-pytorch/blob/main/examples/example_image_to_image.py)  or [04_mnist-image-to-image.ipynb](https://github.com/tneuer/GAN-pytorch/blob/main/notebooks/04_mnist-image-to-image.ipynb)).

Models can either be passed as `torch.nn.Sequential` objects or by defining custom architectures, see [example_input_formats.py](https://github.com/tneuer/GAN-pytorch/blob/main/examples/example_input_formats.py).

Also look at the [jupyter notebooks](https://github.com/tneuer/GAN-pytorch/tree/main/notebooks) for better visualized examples and how to use the library.

#### Supervised / Conditional Learning

```python
import torch
import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading
from vegans.GAN import ConditionalWassersteinGAN
from sklearn.preprocessing import OneHotEncoder # Download sklearn

datapath =  "./data/mnist/"
X_train, y_train, X_test, y_test = (
    loading.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
)
X_train = X_train.reshape((-1, 1, 32, 32)) # required shape
X_test = X_test.reshape((-1, 1, 32, 32))
one_hot_encoder = OneHotEncoder(sparse=False)
y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

x_dim = X_train.shape[1:] # [nr_channels, height, width]
y_dim = y_train.shape[1:]
z_dim = 64

# Define your own architectures here. You can use a Sequential model or an object
# inheriting from torch.nn.Module.
generator = loading.load_example_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
critic = loading.load_example_adversariat(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic")

gan = ConditionalWassersteinGAN(
    generator=generator, adversariat=critic,
    z_dim=z_dim, x_dim=x_dim, y_dim=y_dim,
    folder=None, # optional
    optim={"Generator": torch.optim.RMSprop, "Adversariat": torch.optim.Adam}, # optional
    optim_kwargs={"Generator": {"lr": 0.0001}, "Adversariat": {"lr": 0.0001}}, # optional
    fixed_noise_size=32, # optional
    clip_val=0.01, # optional
    device=None, # optional
    ngpu=0 # optional

)
gan.summary() # optional, shows architecture
gan.fit(
    X_train, y_train, X_test, y_test,
    epochs=5, # optional
    batch_size=32, # optional
    steps={"Generator": 1, "Adversariat": 5}, # optional
    print_every="0.1e", # optional
    save_model_every=None, # optional
    save_images_every=None, # optional
    save_losses_every="0.1e", # optional
    enable_tensorboard=False # optional
)

# Vizualise results
images, losses = gan.get_training_results()
images = images.reshape(-1, *images.shape[2:]) # remove nr_channels for plotting
utils.plot_images(images, labels=np.argmax(gan.fixed_labels.cpu().numpy(), axis=1))
utils.plot_losses(losses)
```

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
  - InfoGAN
  - CycleGAN
  - BEGAN
  - WassersteinGAN SpectralNorm
  - DiscoGAN
- Layers 
  - Inception
  - Residual Block
  - Minibatch discrimination
- Other

  - New links to correct github files

  - Architectures that at least work for mnist

    - Images to compare algorithms
  - Note number params / training time
  
  - Update tests / notebooks
  
  - Include well defined loaders for
  
    - CelebA
    - Mnist
    - Pix2Pix 
    - Map translation
    - ImageNet
    
  - Feature loss (using forward hooks described [here](https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6))
  
    - ```python
      activation = {}
      def the_hook(model, input, output):
          activation["This"] = output.detach()
      
       adversariat.feature_part.register_forward_hook(the_hook)
       print(adversariat(torch.from_numpy(X_train[:25]).float()).shape)
       print(activation["This"].shape)
      ```
  
  - Do not save Discriminator
  
    



- Done
  - ~~enable Wasserstein loss for all architectures (when it makes sense)~~
  - ~~Better default folder (probably None or make current subfolder)~~
  - ~~Better number of default steps for critic~~
  - ~~Adversarial Autoencoder~~
  - ~~get_number_params()~~
  - ~~BicycleGAN~~
  - ~~Introduce latent_space_net and real_space_net to make VAE abstraction better~~
  - ~~VAEGAN~~
  - ~~VAE~~
  - ~~KLGAN~~
  - ~~EBGAN~~
  - ~~GIF the results~~
  - ~~Better use case in README file~~
  - ~~Import good architectures (probably with help of torch)~~
  - ~~Write tests~~
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











