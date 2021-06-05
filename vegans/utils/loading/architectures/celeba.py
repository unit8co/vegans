import torch
import pickle

import numpy as np
import torch.nn as nn

from vegans.utils import get_input_dim
from vegans.utils.layers import LayerReshape, LayerPrintSize


class MyGenerator(nn.Module):
    def __init__(self, x_dim, gen_in_dim):
        super().__init__()
        if len(gen_in_dim) == 1:
            out_shape = (128, 8, 8)
            self.linear_part = nn.Sequential(
                nn.Linear(in_features=gen_in_dim[0], out_features=1024),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=1024, out_features=np.prod(out_shape)),
                nn.LeakyReLU(0.1),
                LayerReshape(shape=out_shape)
            )
            gen_in_dim = out_shape
        else:
            self.linear_part = nn.Identity()

        self.hidden_part = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_in_dim[0], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1),
        )

        desired_output = x_dim[1]
        current_output = gen_in_dim[1]
        in_channels = 128
        i = 3

        while current_output != desired_output:
            out_channels = in_channels // 2
            current_output *= 2
            if current_output != desired_output:
                self.hidden_part.add_module("ConvTraspose{}".format(i), nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1
                    )
                )
                self.hidden_part.add_module("Batchnorm{}".format(i), nn.BatchNorm2d(num_features=out_channels))
                self.hidden_part.add_module("LeakyRelu{}".format(i), nn.LeakyReLU(0.1))

            else: # Last layer
                self.hidden_part.add_module("ConvTraspose{}".format(i), nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=3, kernel_size=4, stride=2, padding=1
                    )
                )
            in_channels = in_channels // 2
            i += 1
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_part(x)
        x = self.hidden_part(x)
        return self.output(x)

def load_celeba_generator(x_dim, z_dim, y_dim=None):
    """ Load some celeba architecture for the generator.

    Parameters
    ----------
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
    if len(z_dim) == 3:
        assert z_dim[1] % 2 == 0, "z_dim[1] must be divisible by 2. Given: {}.".format(z_dim[1])
        assert x_dim[1] % 2 == 0, "`x_dim[1]` must be divisible by 2. Given: {}.".format(x_dim[1])
        assert x_dim[1] % z_dim[1] == 0, "`x_dim[1]` must be divisible by `z_dim[1]`. Given: {} and {}.".format(x_dim[1], z_dim[1])
        assert (x_dim[1] / z_dim[1]) % 2 == 0, "`x_dim[1]/z_dim[1]` must be divisible by 2. Given: {} and {}.".format(x_dim[1], z_dim[1])
        assert z_dim[1] == z_dim[2], "`z_dim[1]` must be equal to `z_dim[2]`. Given: {} and {}.".format(z_dim[1], z_dim[2])

    gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim) if y_dim is not None else z_dim

    return MyGenerator(x_dim=x_dim, gen_in_dim=gen_in_dim)


class MyAdversary(nn.Module):
    def __init__(self, adv_in_dim, last_layer_activation):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Conv2d(in_channels=adv_in_dim[0], out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
        )
        while True:
            current_output = self.hidden_part(torch.randn(size=(2, *adv_in_dim))).shape
            if np.prod(current_output) > 10000:
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.ReLU(),)
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.MaxPool2d(kernel_size=4, stride=2, padding=1))
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.BatchNorm2d(num_features=256))
            else:
                self.hidden_part.add_module(str(len(self.hidden_part) + 1 ), nn.Flatten())
                current_output = self.hidden_part(torch.randn(size=(2, *adv_in_dim))).shape
                break

        self.linear_part = nn.Sequential(
            nn.Linear(in_features=current_output[1], out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1),
        )
        self.output = last_layer_activation()

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.linear_part(x)
        return self.output(x)

def load_celeba_adversary(x_dim, y_dim=None, adv_type="Critic"):
    """ Load some celeba architecture for the adversary.

    Parameters
    ----------
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for adversary.
    """
    possible_types = ["Discriminator", "Critic"]
    if adv_type == "Critic":
        last_layer_activation = nn.Identity
    elif adv_type == "Discriminator":
        last_layer_activation = nn.Sigmoid
    else:
        raise ValueError("'adv_type' must be one of: {}.".format(possible_types))

    adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim) if y_dim is not None else x_dim
    return MyAdversary(adv_in_dim=adv_in_dim, last_layer_activation=last_layer_activation)


class MyEncoder(nn.Module):
    def __init__(self, enc_in_dim, z_dim):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Conv2d(in_channels=enc_in_dim[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )
        sample_input = torch.rand([2, *enc_in_dim])
        flattened_nodes = tuple(self.hidden_part(sample_input).shape)[1]
        self.linear = nn.Linear(in_features=flattened_nodes, out_features=np.prod(z_dim))
        self.reshape = LayerReshape(shape=z_dim)
        self.output = nn.Identity()

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.linear(x)
        x = self.reshape(x)
        return self.output(x)

def load_celeba_encoder(x_dim, z_dim, y_dim=None):
    """ Load some celeba architecture for the encoder.

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
    enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim) if y_dim is not None else x_dim
    return MyEncoder(enc_in_dim=enc_in_dim, z_dim=z_dim)


class MyDecoder(nn.Module):
    def __init__(self, x_dim, dec_in_dim):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=np.prod(dec_in_dim), out_features=np.prod([1, 8, 8])),
            LayerReshape(shape=[1, 8, 8]),
        )
        desired_output = x_dim[1]
        current_output = 8
        in_channels = 1
        i = 2

        while current_output != desired_output:
            out_channels = in_channels * 2
            current_output *= 2
            if current_output != desired_output:
                self.hidden_part.add_module("ConvTraspose{}".format(i), nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1
                    )
                )
                self.hidden_part.add_module("Batchnorm{}".format(i), nn.BatchNorm2d(num_features=out_channels))
                self.hidden_part.add_module("LeakyRelu{}".format(i), nn.LeakyReLU(0.1))

            else: # Last layer
                self.hidden_part.add_module("ConvTraspose{}".format(i), nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=3, kernel_size=4, stride=2, padding=1
                    )
                )
            in_channels = in_channels * 2
            i += 1
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_celeba_decoder(x_dim, z_dim, y_dim=None):
    """ Load some mnist architecture for the decoder.

    Parameters
    ----------
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for decoder.
    """
    assert x_dim[1] % 2 == 0, "`x_dim[1]` must be divisible by 2. Given: {}.".format(x_dim[1])
    assert x_dim[1] % 8 == 0, "`x_dim[1]` must be divisible by 8. Given: {}.".format(x_dim[1])
    assert (x_dim[1] / 8) % 2 == 0, "`x_dim[1]/8` must be divisible by 2. Given: {}.".format(x_dim[1])
    dec_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim) if y_dim is not None else z_dim
    return MyDecoder(x_dim=x_dim, dec_in_dim=dec_in_dim)