import torchvision

from vegans.utils.architectures.mnist import (
    preprocess_mnist, load_mnist_generator, load_mnist_adversary,
    load_mnist_encoder, load_mnist_decoder, load_mnist_autoencoder
)
from vegans.utils.architectures.example import (
    load_example_generator, load_example_adversary, load_example_encoder,
    load_example_decoder, load_example_autoencoder
)
from vegans.utils.architectures.celeba import (
    preprocess_celeba
)
from vegans.utils.architectures.cifar import (
    preprocess_cifar
)

def load_data(root, which=None, **kwargs):
    """ Wrapper around torchvision.datasets with certain preprocessing steps

    So far available are:
        - MNIST: Handwritten digits with labels. Can be downloaded via `download=True`.
        - FashionMNIST: Clothes with labels. Can be downloaded via `download=True`.
        - CelebA: Pictures of celebrities with attributes. Must be downloaded from https://www.kaggle.com/jessicali9530/celeba-dataset
                  and moved into `root` folder.
        - CIFAR: Pictures of objects with labels. Must be downloaded from http://www.cs.toronto.edu/~kriz/cifar.html
                  and moved into `root` folder.

    Parameters
    ----------
    root : str
        Path to root directory. Is created if `download=True` and the folder does not exists yet.
    which : str, optional
        One of the torchvision.datasets.
    **kwargs
        Keyword arguments to torchvision.datasets (`https://pytorch.org/vision/0.8/datasets.html`).

    Returns
    -------
    np.array
        Numpy array or torch dataset with train and test data.
    """
    available = ["MNIST", "FashionMNIST", "CelebA", "CIFAR"]
    which = which.replace("mnist", "MNIST")

    if which.lower() == "mnist":
        loader = eval("torchvision.datasets." + which)
        torch_data_train = loader(root=root, train=True, **kwargs)
        torch_data_test = loader(root=root, train=False, **kwargs)
        X_train, y_train = preprocess_mnist(torch_data_train, normalize=True, pad=2)
        X_test, y_test = preprocess_mnist(torch_data_test, normalize=True, pad=2)
        return X_train, y_train, X_test, y_test
    elif which.lower() == "fashionmnist":
        loader = eval("torchvision.datasets." + which)
        torch_data_train = loader(root=root, train=True, **kwargs)
        torch_data_test = loader(root=root, train=False, **kwargs)
        X_train, y_train = preprocess_mnist(torch_data_train, normalize=True, pad=2)
        X_test, y_test = preprocess_mnist(torch_data_test, normalize=True, pad=2)
        return X_train, y_train, X_test, y_test
    elif which.lower() == "celeba":
        train_dataloader = preprocess_celeba(root=root, **kwargs)
        return train_dataloader
    elif which.lower() == "cifar":
        X_train, y_train, X_test, y_test = preprocess_cifar(root=root, normalize=True, pad=0)
        return X_train, y_train, X_test, y_test
    else:
        raise ValueError("`which` must be one of {}.".format(available))

def load_generator(x_dim, z_dim, y_dim=None, which="example"):
    """ Load pre-defined (**NOT** pre-trained) architecture for a generator.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.
    which : str, optional
        Currently one of ["example", "mnist"]. Specifying "example" will provide you
        with a minimally working architecture for most use cases. However it's generic definition
        and underpowered structure will probably not result in desirable results. "mnist" provides
        you with a working architecture (depending of course on the choice of other hyper-parameters like optimizer)
        for both "mnist" datasets (MNIST and FashionMNIST). It might be useful for other problems where the input images
        are of the form (1, 32, 32) but it is not guaranteed. It's more powerful architecture might some take to train
        but should lead to reasonable results for certain use cases.

    Returns
    -------
    nn.Module
        Generator architecture that can be passed to any GAN algorithm.
    """
    available = ["example", "mnist"]
    if which == "example":
        return load_example_generator(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_generator(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_adversary(x_dim, z_dim, y_dim=None, adv_type="Critic", which="example"):
    """ Load pre-defined (**NOT** pre-trained) architecture for a adversary.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.
    which : str, optional
        Currently one of ["example", "mnist"]. Specifying "example" will provide you
        with a minimally working architecture for most use cases. However it's generic definition
        and underpowered structure will probably not result in desirable results. "mnist" provides
        you with a working architecture (depending of course on the choice of other hyper-parameters like optimizer)
        for both "mnist" datasets (MNIST and FashionMNIST). It might be useful for other problems where the input images
        are of the form (1, 32, 32) but it is not guaranteed. It's more powerful architecture might some take to train
        but should lead to reasonable results for certain use cases.

    Returns
    -------
    nn.Module
        Adversary architecture that can be passed to any GAN algorithm.
    """
    available = ["example", "mnist"]
    if which == "example":
        return load_example_adversary(x_dim, z_dim, y_dim=y_dim, adv_type=adv_type)
    elif which == "mnist":
        return load_mnist_adversary(x_dim, z_dim, y_dim=y_dim, adv_type=adv_type)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_encoder(x_dim, z_dim, y_dim=None, which="example"):
    """ Load pre-defined (**NOT** pre-trained) architecture for an encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.
    which : str, optional
        Currently one of ["example", "mnist"]. Specifying "example" will provide you
        with a minimally working architecture for most use cases. However it's generic definition
        and underpowered structure will probably not result in desirable results. "mnist" provides
        you with a working architecture (depending of course on the choice of other hyper-parameters like optimizer)
        for both "mnist" datasets (MNIST and FashionMNIST). It might be useful for other problems where the input images
        are of the form (1, 32, 32) but it is not guaranteed. It's more powerful architecture might some take to train
        but should lead to reasonable results for certain use cases.

    Returns
    -------
    nn.Module
        Encoder architecture that can be passed to certain GAN algorithms.
    """
    available = ["example", "mnist"]
    if which == "example":
        return load_example_encoder(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_encoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_decoder(x_dim, z_dim, y_dim=None, which="example"):
    """ Load pre-defined (**NOT** pre-trained) architecture for a decoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.
    which : str, optional
        Currently one of ["example", "mnist"]. Specifying "example" will provide you
        with a minimally working architecture for most use cases. However it's generic definition
        and underpowered structure will probably not result in desirable results. "mnist" provides
        you with a working architecture (depending of course on the choice of other hyper-parameters like optimizer)
        for both "mnist" datasets (MNIST and FashionMNIST). It might be useful for other problems where the input images
        are of the form (1, 32, 32) but it is not guaranteed. It's more powerful architecture might some take to train
        but should lead to reasonable results for certain use cases.

    Returns
    -------
    nn.Module
        Decoder architecture that can be passed to some GAN algorithms and VAEs.
    """
    available = ["example", "mnist"]
    if which == "example":
        return load_example_decoder(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_decoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_autoencoder(x_dim, z_dim, y_dim=None, which="example"):
    """ Load pre-defined (**NOT** pre-trained) architecture for an auto-encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.
    which : str, optional
        Currently one of ["example", "mnist"]. Specifying "example" will provide you
        with a minimally working architecture for most use cases. However it's generic definition
        and underpowered structure will probably not result in desirable results. "mnist" provides
        you with a working architecture (depending of course on the choice of other hyper-parameters like optimizer)
        for both "mnist" datasets (MNIST and FashionMNIST). It might be useful for other problems where the input images
        are of the form (1, 32, 32) but it is not guaranteed. It's more powerful architecture might some take to train
        but should lead to reasonable results for certain use cases.

    Returns
    -------
    nn.Module
        Auto-encoder architecture that can be passed to for example the EBGAN.
    """
    available = ["example", "mnist"]
    if which == "example":
        return load_example_autoencoder(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_autoencoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

