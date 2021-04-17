import torch
import pytest

import numpy as np

from vegans.GAN import (
    EBGAN,
    KLGAN,
    LSGAN,
    VanillaGAN,
    WassersteinGAN,
    WassersteinGANGP
)
from vegans.utils.layers import LayerReshape

gans = [
    KLGAN,
    LSGAN,
    VanillaGAN,
    WassersteinGAN,
    WassersteinGANGP
]
last_layers = [
    torch.nn.Sigmoid,
    torch.nn.Sigmoid,
    torch.nn.Sigmoid,
    torch.nn.Identity,
    torch.nn.Identity
]
optimizers = [
    torch.optim.Adam,
    torch.optim.Adam,
    torch.optim.Adam,
    torch.optim.RMSprop,
    torch.optim.RMSprop,
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

    for (gan, last_layer) in zip(gans, last_layers):
        adv = generate_net(in_dim=16, last_layer=last_layer, out_dim=1)

        testgan = gan(generator=gen, adversariat=adv, x_dim=16, z_dim=10, folder=None)
        names = [key for key, _ in testgan.optimizers.items()]
        assert ("Generator" in names) and ("Adversariat" in names)
        with pytest.raises(TypeError):
            gan(generator=gen, adversariat=adv, x_dim=17, z_dim=10, folder=None)
        with pytest.raises(TypeError):
            gan(generator=gen, adversariat=adv, x_dim=16, z_dim=11, folder=None)

    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=17)
    for (gan, last_layer) in zip(gans, last_layers):
        adv = generate_net(in_dim=16, last_layer=last_layer, out_dim=1)

        with pytest.raises(AssertionError):
            gan(generator=gen, adversariat=adv, x_dim=16, z_dim=10, folder=None)

def test_fit_vector():
    X_train = np.zeros(shape=[100, 16])
    X_test = np.zeros(shape=[100, 16])

    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=16)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    for (gan, last_layer) in zip(gans, last_layers):
        adv = generate_net(in_dim=16, last_layer=last_layer, out_dim=1)

        testgan = gan(generator=gen, adversariat=adv, x_dim=16, z_dim=10, folder=None)
        testgan.fit(
            X_train=X_train, X_test=X_test, **fit_kwargs
        )
        assert hasattr(testgan, "logged_losses")

def test_fit_error_vector():
    X_train = np.zeros(shape=[100, 16])
    X_train_wrong_shape1 = np.zeros(shape=[100])
    X_train_wrong_shape2 = np.zeros(shape=[100, 17])
    X_train_wrong_shape3 = np.zeros(shape=[100, 17, 14])
    X_train_wrong_shape4 = np.zeros(shape=[100, 17, 10, 10])
    X_test_wrong_shape = np.zeros(shape=[100, 17])

    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=16)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    for (gan, last_layer) in zip(gans, last_layers):
        adv = generate_net(in_dim=16, last_layer=last_layer, out_dim=1)

        testgan = gan(generator=gen, adversariat=adv, x_dim=16, z_dim=10, folder=None)
        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape1, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape2, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape3, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape4, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train, X_test=X_test_wrong_shape, **fit_kwargs
            )

def test_fit_image():
    im_shape = [3, 16, 16]
    X_train = np.zeros(shape=[100, *im_shape])
    X_test = np.zeros(shape=[100, *im_shape])

    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=im_shape)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    for (gan, last_layer) in zip(gans, last_layers):
        adv = generate_net(in_dim=im_shape, last_layer=last_layer, out_dim=1)

        testgan = gan(generator=gen, adversariat=adv, x_dim=im_shape, z_dim=10, folder=None)
        testgan.fit(
            X_train=X_train, X_test=X_test, **fit_kwargs
        )
        assert hasattr(testgan, "logged_losses")

def test_fit_error_image():
    im_shape = [3, 16, 16]
    X_train = np.zeros(shape=[100, *im_shape])
    X_train_wrong_shape1 = np.zeros(shape=[100])
    X_train_wrong_shape2 = np.zeros(shape=[100, 17])
    X_train_wrong_shape3 = np.zeros(shape=[100, 17, 14])
    X_train_wrong_shape4 = np.zeros(shape=[100, 4, 16, 16])
    X_test_wrong_shape = np.zeros(shape=[100, 3, 17, 17])

    gen = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid, out_dim=im_shape)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    for (gan, last_layer) in zip(gans, last_layers):
        adv = generate_net(in_dim=im_shape, last_layer=last_layer, out_dim=1)

        testgan = gan(generator=gen, adversariat=adv, x_dim=im_shape, z_dim=10, folder=None)
        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape1, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape2, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape3, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train_wrong_shape4, **fit_kwargs
            )

        with pytest.raises(AssertionError):
            testgan.fit(
                X_train=X_train, X_test=X_test_wrong_shape, **fit_kwargs
            )


def test_default_optimizers():
    for gan, optim in zip(gans, optimizers):
        assert gan._default_optimizer(gan) == optim
