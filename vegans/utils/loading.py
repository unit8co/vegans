import pickle

import numpy as np

from vegans.utils.layers import LayerReshape
from vegans.utils.architectures.mnist import (
    load_mnist_generator, load_mnist_adversariat, load_mnist_encoder,
    load_mnist_decoder, load_mnist_autoencoder
)
from vegans.utils.architectures.example import (
    load_example_generator, load_example_adversariat, load_example_encoder,
    load_example_decoder, load_example_autoencoder
)

def load_mnist(datapath, normalize=True, pad=None, return_datasets=False):
    """ Load the mnist data from datapath.

    Parameters
    ----------
    datapath : TYPE
        Path to the train and test image pickle files.
    normalize : bool, optional
        If True, data will be scaled to the interval [0, 1]
    pad : None, optional
        Integer indicating the padding applied to each side of the input images.
    return_datasets : bool, optional
        If True, a vegans.utils.DataSet is returned which can be passed to a torch.DataLoader

    Returns
    -------
    numpy.array, vegans.utils.DataSet
        train and test images as well as labels.
    """
    datapath = datapath if datapath.endswith("/") else datapath+"/"
    with open(datapath+"train_images.pickle", "rb") as f:
        X_train, y_train = pickle.load(f)
    with open(datapath+"test_images.pickle", "rb") as f:
        X_test, y_test = pickle.load(f)

    if normalize:
        max_number = X_train.max()
        X_train = X_train / max_number
        X_test = X_test / max_number

    if pad:
        X_train = np.pad(X_train, [(0, 0), (pad, pad), (pad, pad)], mode='constant')
        X_test = np.pad(X_test, [(0, 0), (pad, pad), (pad, pad)], mode='constant')

    if return_datasets:
        train = DataSet(X_train, y_train)
        test = DataSet(X_test, y_test)
        return(train, test)
    return X_train, y_train, X_test, y_test


def load_generator(x_dim, z_dim, y_dim=None, method="example"):
    available_methods = ["example", "mnist"]
    if method == "example":
        return load_example_generator(x_dim, z_dim, y_dim=y_dim)
    elif method == "mnist":
        return load_mnist_generator(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`method` must be one of {}. Given: {}.".format(available_methods, method))

def load_adversariat(x_dim, z_dim, y_dim=None, adv_type="Critic", method="example"):
    available_methods = ["example", "mnist"]
    if method == "example":
        return load_example_adversariat(x_dim, z_dim, y_dim=y_dim, adv_type=adv_type)
    elif method == "mnist":
        return load_mnist_adversariat(x_dim, z_dim, y_dim=y_dim, adv_type=adv_type)
    else:
        raise ValueError("`method` must be one of {}. Given: {}.".format(available_methods, method))

def load_encoder(x_dim, z_dim, y_dim=None, method="example"):
    available_methods = ["example", "mnist"]
    if method == "example":
        return load_example_encoder(x_dim, z_dim, y_dim=y_dim)
    elif method == "mnist":
        return load_mnist_encoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`method` must be one of {}. Given: {}.".format(available_methods, method))

def load_decoder(x_dim, z_dim, y_dim=None, method="example"):
    available_methods = ["example", "mnist"]
    if method == "example":
        return load_example_decoder(x_dim, z_dim, y_dim=y_dim)
    elif method == "mnist":
        return load_mnist_decoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`method` must be one of {}. Given: {}.".format(available_methods, method))

def load_autoencoder(x_dim, z_dim, y_dim=None, method="example"):
    available_methods = ["example", "mnist"]
    if method == "example":
        return load_example_autoencoder(x_dim, z_dim, y_dim=y_dim)
    elif method == "mnist":
        return load_mnist_autoencoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`method` must be one of {}. Given: {}.".format(available_methods, method))

