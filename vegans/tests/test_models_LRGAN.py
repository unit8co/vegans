import torch
import pytest

import numpy as np

from vegans.GAN import LRGAN
from vegans.utils.layers import LayerReshape

gans = [
    LRGAN,
]
last_layers = [
    torch.nn.Sigmoid,
]
optimizers = [
    torch.optim.Adam,
]

def generate_net(in_dim, last_layer, out_dim):
    class MyNetwork(torch.nn.Module):
        def __init__(self, in_dim, last_layer, out_dim):
            super().__init__()
            in_dim = np.prod(in_dim)
            out_dim = [out_dim] if isinstance(out_dim, int) else out_dim
            self.hidden_part = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(in_dim, 16),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(16, np.prod(out_dim)),
                LayerReshape(out_dim)

            )
            self.output = last_layer()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    return MyNetwork(in_dim, last_layer, out_dim)


def test_init():
    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=16)
    enc = generate_net(in_dim=16, last_layer=torch.nn.Identity, out_dim=10)

    for gan, last_layer in zip(gans, last_layers):
        disc = generate_net(in_dim=16, last_layer=last_layer, out_dim=1)

        testgan = gan(generator=gen, adversariat=disc, encoder=enc, x_dim=16, z_dim=10, folder=None)
        names = [key for key, _ in testgan.optimizers.items()]
        assert ("Generator" in names) and ("Adversariat" in names) and ("Encoder" in names)
        with pytest.raises(TypeError):
            gan(generator=gen, adversariat=disc, encoder=enc, x_dim=17, z_dim=10, folder=None)
        with pytest.raises(TypeError):
            gan(generator=gen, adversariat=disc, encoder=enc, x_dim=16, z_dim=11, folder=None)

    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=17)
    for gan, last_layer in zip(gans, last_layers):
        with pytest.raises(AssertionError):
            gan(generator=gen, adversariat=disc, encoder=enc, x_dim=16, z_dim=10, folder=None)

    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=16)
    enc = generate_net(in_dim=16, last_layer=torch.nn.Identity, out_dim=17)
    for gan, last_layer in zip(gans, last_layers):
        with pytest.raises(AssertionError):
            gan(generator=gen, adversariat=disc, encoder=enc, x_dim=16, z_dim=10, folder=None)

def test_default_optimizers():
    for gan, optim in zip(gans, optimizers):
        assert gan._default_optimizer(gan) == optim
