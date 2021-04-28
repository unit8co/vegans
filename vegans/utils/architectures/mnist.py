import pickle

import numpy as np
import torch.nn as nn

from vegans.utils.utils import get_input_dim
from vegans.utils.layers import LayerReshape, LayerPrintSize, LayerInception, LayerResidualConvBlock


def preprocess_mnist(torch_data, normalize=True, pad=None):
    """Load the mnist data from root.

    Parameters
    ----------
    torch_data : dict
        Original data loaded by `tochvision.datasets`
    normalize : bool, optional
        If True, data will be scaled to the interval [0, 1]
    pad : int, optional
        Integer indicating the padding applied to each side of the input images.

    Returns
    -------
    numpy.array
        train and test data as well as labels.
    """
    X = torch_data.data.numpy()
    y = torch_data.targets.numpy()

    if normalize:
        max_number = X.max()
        X = X / max_number

    if pad:
        X = np.pad(X, [(0, 0), (pad, pad), (pad, pad)], mode='constant')

    return X, y


class MyGenerator(nn.Module):
    def __init__(self, gen_in_dim, x_dim):
        super().__init__()

        if len(gen_in_dim) == 1:
            self.prepare = nn.Sequential(
                nn.Linear(in_features=gen_in_dim[0], out_features=256),
                LayerReshape(shape=[1, 16, 16])
            )
            nr_channels = 1
        else:
            current_dim = z_dim[1]
            nr_channels = gen_in_dim[0]
            self.prepare = []
            while current_dim < 16:
                self.prepare.append(nn.ConvTranspose2d(
                    in_channels=nr_channels, out_channels=5, kernel_size=4, stride=2, padding=1
                    )
                )
                nr_channels = 5
                current_dim *= 2
            self.prepare = nn.Sequential(*self.prepare)

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
        x = self.prepare(x)
        x = self.encoding(x)
        x = self.decoding(x)
        return self.output(x)

def load_mnist_generator(x_dim, z_dim, y_dim=None):
    """ Load some mnist architecture for the generator.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for generator,.
    """
    z_dim = [z_dim] if isinstance(z_dim, int) else z_dim
    y_dim = tuple([y_dim]) if isinstance(y_dim, int) else y_dim
    assert y_dim is None or y_dim == (10, ) or y_dim == (20, ), "y_dim must be (10, ). Given: {}.".format(y_dim)
    if len(z_dim) > 1:
        assert (z_dim[1] <= 16) and (z_dim[1] % 2 == 0), "z_dim[1] must be smaller 16 and divisible by 2. Given: {}.".format(z_dim[1])
        assert z_dim[1] == z_dim[2], "z_dim[1] must be equal to z_dim[2]. Given: {} and {}.".format(z_dim[1], z_dim[2])

    if y_dim is not None:
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
    else:
        gen_in_dim = z_dim

    return MyGenerator(gen_in_dim=gen_in_dim, x_dim=x_dim)


class MyAdversary(nn.Module):
    def __init__(self, adv_in_dim, last_layer):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Conv2d(in_channels=adv_in_dim[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.output = last_layer()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_mnist_adversary(x_dim, z_dim, y_dim=None, adv_type="Critic"):
    """ Load some mnist architecture for the adversary.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for adversary.
    """
    possible_types = ["Discriminator", "Critic"]
    if adv_type == "Critic":
        last_layer = nn.Identity
    elif adv_type == "Discriminator":
        last_layer = nn.Sigmoid
    else:
        raise ValueError("'adv_type' must be one of: {}.".format(possible_types))

    if len(x_dim) == 3:
        assert x_dim == (1, 32, 32), "x_dim must be (1, 32, 32). Given: {}.".format(x_dim)
    assert y_dim is None or y_dim == (10, ), "y_dim must be (10, ). Given: {}.".format(y_dim)

    if y_dim is not None:
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        adv_in_dim = x_dim

    return MyAdversary(adv_in_dim=adv_in_dim, last_layer=last_layer)


class MyEncoder(nn.Module):
    def __init__(self, enc_in_dim, z_dim):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Conv2d(in_channels=enc_in_dim[0], out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=np.prod(z_dim))
        )
        self.output = nn.Identity()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_mnist_encoder(x_dim, z_dim, y_dim=None):
    """ Load some mnist architecture for the encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for encoder.
    """
    z_dim = [z_dim] if isinstance(z_dim, int) else z_dim
    assert x_dim == (1, 32, 32) or x_dim == (11, 32, 32), "x_dim must be (1, 32, 32). Given: {}.".format(x_dim)
    assert y_dim is None or y_dim == (10, ), "y_dim must be (10, ). Given: {}.".format(y_dim)
    assert len(z_dim) == 1, "z_dim must be of length one. Given: {}.".format(z_dim)

    if y_dim is not None:
        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        enc_in_dim = x_dim

    return MyEncoder(enc_in_dim=enc_in_dim, z_dim=z_dim)


class MyDecoder(nn.Module):
    def __init__(self, dec_in_dim):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Linear(in_features=np.prod(dec_in_dim), out_features=np.prod([1, 8, 8])),
            LayerReshape(shape=[1, 8, 8]),
            nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_mnist_decoder(x_dim, z_dim, y_dim=None):
    """ Load some mnist architecture for the decoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for decoder.
    """
    assert x_dim == (1, 32, 32), "x_dim must be (1, 32, 32). Given: {}.".format(x_dim)
    assert y_dim is None or y_dim == (10, ), "y_dim must be (10, ). Given: {}.".format(y_dim)

    if y_dim is not None:
        dec_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
    else:
        dec_in_dim = z_dim

    return MyDecoder(dec_in_dim=dec_in_dim)


class MyAutoEncoder(nn.Module):
    def __init__(self, ae_in_dim):
        super().__init__()
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels=ae_in_dim[0], out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
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

def load_mnist_autoencoder(x_dim, z_dim, y_dim=None):
    """ Load some mnist architecture for the auto-encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for autoencoder.
    """
    assert x_dim == (1, 32, 32), "x_dim must be (1, 32, 32). Given: {}.".format(x_dim)
    assert y_dim is None or y_dim == (10, ), "y_dim must be (10, ). Given: {}.".format(y_dim)

    if y_dim is not None:
        ae_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        ae_in_dim = x_dim

    return MyAutoEncoder(ae_in_dim=ae_in_dim)