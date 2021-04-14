import torch
import pickle

import numpy as np
import torch.nn as nn

from vegans.utils.utils import get_input_dim
from vegans.utils.layers import LayerReshape

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


def load_example_architectures(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for generator, adversariat and encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : TYPE
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for example generator, adversariat and encoder.
    """
    generator = load_example_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    adversariat = load_example_adversariat(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    encoder = load_example_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    return generator, adversariat, encoder


def load_example_generator(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for the generator.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : TYPE
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for generator,.
    """
    if y_dim is not None:
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
    else:
        gen_in_dim = z_dim

    class MyGenerator(nn.Module):
        def __init__(self, z_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(gen_in_dim), 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, int(np.prod(x_dim))),
                LayerReshape(x_dim)
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden_part(x)
            return self.output(x)

    return MyGenerator(z_dim=z_dim)


def load_example_adversariat(x_dim, z_dim, y_dim=None, adv_type="Critic"):
    """ Load some example architecture for the adversariat.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : TYPE
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for adversariat.
    """
    possible_types = ["Discriminator", "Critic"]
    if adv_type == "Critic":
        last_layer = nn.Identity
    elif adv_type == "Discriminator":
        last_layer = nn.Sigmoid
    else:
        raise ValueError("'adv_type' must be one of: {}.".format(possible_types))

    if y_dim is not None:
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        adv_in_dim = x_dim

    class MyAdversariat(nn.Module):
        def __init__(self, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(adv_in_dim), 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1)
            )
            self.output = last_layer()

        def forward(self, x):
            x = self.hidden_part(x)
            return self.output(x)

    return MyAdversariat(x_dim=x_dim)


def load_example_encoder(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for the encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : TYPE
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for encoder.
    """
    z_dim = [z_dim] if isinstance(z_dim, int) else z_dim
    class MyEncoder(nn.Module):
        def __init__(self, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(x_dim), 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, np.prod(z_dim)),
                LayerReshape(z_dim)
            )
            self.output = nn.Identity()

        def forward(self, x):
            x = self.hidden_part(x)
            return self.output(x)

    return MyEncoder(x_dim=x_dim)


def load_example_decoder(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for the decoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : TYPE
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for decoder.
    """
    x_dim = [x_dim] if isinstance(x_dim, int) else x_dim
    class MyDecoder(nn.Module):
        def __init__(self, z_dim, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(z_dim), 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, np.prod(x_dim)),
                LayerReshape(x_dim)
            )
            self.output = nn.Identity()

        def forward(self, x):
            x = self.hidden_part(x)
            return self.output(x)

    return MyDecoder(x_dim=x_dim, z_dim=z_dim)


def load_example_autoencoder(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for the auto-encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : TYPE
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for autoencoder.
    """
    if y_dim is not None:
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        adv_in_dim = x_dim

    class MyAutoEncoder(nn.Module):
        def __init__(self, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(adv_in_dim), 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, np.prod(x_dim)),
                LayerReshape(x_dim)
            )
            self.output = nn.Identity()

        def forward(self, x):
            x = self.hidden_part(x)
            return self.output(x)

    return MyAutoEncoder(x_dim=x_dim)