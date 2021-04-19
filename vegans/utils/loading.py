import pickle

import numpy as np

from vegans.utils.layers import LayerReshape
from vegans.utils.architectures.mnist import (
    load_mnist_data, load_mnist_generator, load_mnist_adversariat,
    load_mnist_encoder, load_mnist_decoder, load_mnist_autoencoder
)
from vegans.utils.architectures.example import (
    load_example_generator, load_example_adversariat, load_example_encoder,
    load_example_decoder, load_example_autoencoder
)

def load_data(datapath, which="example"):
    available = ["example", "mnist"]
    if which == "example":
        return load_mnist_data(datapath=datapath, normalize=True, pad=None, return_datasets=False)
    elif which == "mnist":
        return load_mnist_data(datapath=datapath, normalize=True, pad=2, return_datasets=False)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))


def load_generator(x_dim, z_dim, y_dim=None, which="example"):
    available = ["example", "mnist"]
    if which == "example":
        return load_example_generator(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_generator(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_adversariat(x_dim, z_dim, y_dim=None, adv_type="Critic", which="example"):
    available = ["example", "mnist"]
    if which == "example":
        return load_example_adversariat(x_dim, z_dim, y_dim=y_dim, adv_type=adv_type)
    elif which == "mnist":
        return load_mnist_adversariat(x_dim, z_dim, y_dim=y_dim, adv_type=adv_type)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_encoder(x_dim, z_dim, y_dim=None, which="example"):
    available = ["example", "mnist"]
    if which == "example":
        return load_example_encoder(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_encoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_decoder(x_dim, z_dim, y_dim=None, which="example"):
    available = ["example", "mnist"]
    if which == "example":
        return load_example_decoder(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_decoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

def load_autoencoder(x_dim, z_dim, y_dim=None, which="example"):
    available = ["example", "mnist"]
    if which == "example":
        return load_example_autoencoder(x_dim, z_dim, y_dim=y_dim)
    elif which == "mnist":
        return load_mnist_autoencoder(x_dim, z_dim, y_dim=y_dim)
    else:
        raise ValueError("`which` must be one of {}. Given: {}.".format(available, which))

