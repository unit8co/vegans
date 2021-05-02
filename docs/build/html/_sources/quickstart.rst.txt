Quickstart guide
================

Here we're setting up a quick first example to train on the ``MNIST`` image dataset.

Installation
------------

First we need to install ``vegans``. You can do this either with::

    pip install vegans

or via::

    git clone https://github.com/unit8co/vegans.git
    cd vegans
    pip install -e .

Test if the module can be imported with::

    python -c "import vegans"

Loading data
------------

Here we will quickly load in the data. Dedicated data loaders exist for

- MNIST
- FashionMNIST
- CelebA
- CIFAR10

Only the first two are downloaded automatically. The last two need to be downloaded manually but links to the sources of the data are provided
if the data does not yet exist in the ``root`` directory.

Let's load the ``MNIST`` data with the ``loading`` module::

    root = "./data/"
    X_train, y_train, X_test, y_test = loading.load_data(root=root, which="mnist", download=True)

This downloads the data into ``root`` if it does not yet exist in there. Each image will be of shape ``(32, 32)`` but vegans requires for the usage
of convolutional neural networks a third dimension for the number of channels. This can easily be done with numpy::

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))

To be able to use a conditional network we also need to one-hot encode the labels. This can also be done with numpy (we recommend the usage of scikit-learn, but want to avoid additional dependencies)::

    nb_classes = len(set(y_train))
    y_train = np.eye(nb_classes)[y_train.reshape(-1)]
    y_test = np.eye(nb_classes)[y_test.reshape(-1)]

Now we can start defining our networks.

Model definition
----------------

What kind of networks you need to define depends on which algorithm you use. Mainly there are three different choices:

1. GAN1v1 require
    - Generator
    - Adversary

2. GANGAE require
    - Generator
    - Adversary
    - Encoder

3. VAE require
    - Encoder
    - Decoder

In this guide we will use the ``VanillaGAN`` which belongs to the first category.

We first need to determine the input and output dimensions for all networks. In the unsupervised / unconditional case it is easy:

- Generator
    - Input: ``z_dim`` latent dimension (hyper-parameter)
    - Output: ``x_dim`` image dimension
- Discriminator
    - Input: ``x_dim`` image dimension
    - Output: ``1`` single output node (might also be different)

For the supervised / conditional algorithms it is a just a little bit more difficult:

- c-Generator
    - Input: ``z_dim + y_dim`` latent dimension and label dimension
    - Output: ``x_dim`` image dimension
- c-Discriminator
    - Input: ``x_dim + y_dim`` image dimension and label dimension
    - Output: ``1`` single output node (might also be different)

We can get these sizes with::

    x_dim = X_train.shape[1:]
    y_dim = y_train.shape[1:]
    z_dim = 64

    gen_in_dim = vegans.utils.utils.get_input_dim(z_dim, y_dim)
    adv_in_dim = vegans.utils.utils.get_input_dim(x_dim, y_dim)

The definition of a generator and adversary architecture is without a doubt the most important (and most difficult) part of
GAN training. We will use the following architecture::

    class MyGenerator(nn.Module):
        def __init__(self, gen_in_dim, x_dim):
            super().__init__()

            self.encoding = nn.Sequential(
                nn.Conv2d(in_channels=nr_channels, out_channels=64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(0.2),
            )
            self.decoding = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.encoding(x)
            x = self.decoding(x)
            return self.output(x)

    generator = MyGenerator(gen_in_dim=gen_in_dim, x_dim=x_dim)

Almost the same architecture can be loaded in one line again from the loading module which takes care of choosing the right input dimension::

    generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")
    discriminator = loading.load_adversary(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator", which="mnist")

    gan_model = ConditionalVanillaGAN(
        generator=generator, adversary=discriminator, x_dim=x_dim, z_dim=z_dim, y_dim=y_dim,
    )
    gan_model.summary()

Model training
--------------

The model training can now be done in one line of code::

    gan_model.fit(X_train=X_train, y_train=y_train)

There are quite a few of optional hyper-parameters to choose from in this step. See the full code example below.
The training of the GAN might take a while, depending on the size of your networks, the number of training examples
and your hardware.

Model evaluation
----------------

We can finally investiagte the results of the GAN with::

    samples, losses = gan_model.get_training_results(by_epoch=False)

    fixed_labels = np.argmax(gan_model.get_fixed_labels(), axis=1)
    fig, axs = plot_images(images=samples, labels=fixed_labels, show=False)
    plt.show()

You can also generate examples from now on by providing the labels as input::

    test_labels = np.eye(N=10)
    test_samples = gan_model.generate(y=test_labels)
    fig, axs = plot_images(images=test_samples, labels=np.argmax(test_labels, axis=1))

Saving and loading models
-------------------------

After a network has been trained in can easily be saved with::

    gan_model.save("model.torch")

and later loaded::

    gan_model = VanillaGAN.load("model.torch")

or::

    gan_model = torch.load("model.torch")

Full code snippet
-----------------

This is the previous code in one single block::

    import numpy as np
    import vegans.utils.loading as loading
    from vegans.utils.utils import plot_images
    from vegans.GAN import ConditionalVanillaGAN

    root = "./data/"
    X_train, y_train, X_test, y_test = loading.load_data(root=root, which="mnist", download=True)

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    nb_classes = len(set(y_train))
    y_train = np.eye(nb_classes)[y_train.reshape(-1)]
    y_test = np.eye(nb_classes)[y_test.reshape(-1)]

    x_dim = X_train.shape[1:]
    y_dim = y_train.shape[1:]
    z_dim = 64

    generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")
    discriminator = loading.load_adversary(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator", which="mnist")

    gan_model = ConditionalVanillaGAN(
        generator=generator, adversary=discriminator,
        x_dim=x_dim, z_dim=z_dim, y_dim=y_dim,
        optim=None, optim_kwargs=None,                # Optional
        feature_layer=None,                           # Optional
        fixed_noise_size=32,                          # Optional
        device=None,                                  # Optional
        ngpu=None,                                    # Optional
        folder=None,                                  # Optional
        secure=True                                   # Optional
    )

    gan_model.summary()
    gan_model.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,           # Optional
        y_test=y_test,           # Optional
        batch_size=32,           # Optional
        epochs=2,                # Optional
        steps=None,              # Optional
        print_every="0.2e",      # Optional
        save_model_every=None,   # Optional
        save_images_every=None,  # Optional
        save_losses_every=10,    # Optional
        enable_tensorboard=False # Optional
    )
    samples, losses = gan_model.get_training_results(by_epoch=False)

    fixed_labels = np.argmax(gan_model.get_fixed_labels(), axis=1)
    fig, axs = plot_images(images=samples, labels=fixed_labels)

    test_labels = np.eye(N=10)
    test_samples = gan_model.generate(y=test_labels)
    fig, axs = plot_images(images=test_samples, labels=np.argmax(test_labels, axis=1))