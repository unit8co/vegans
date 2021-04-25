import torch
import pytest

import numpy as np

from vegans.GAN import (
    ConditionalKLGAN,
    ConditionalLSGAN,
    ConditionalVanillaGAN,
    ConditionalWassersteinGAN,
    ConditionalWassersteinGANGP,
    ConditionalPix2Pix
)
from vegans.utils.utils import get_input_dim
from vegans.utils.layers import LayerReshape

networks = [
    (ConditionalKLGAN, torch.nn.Sigmoid),
    (ConditionalLSGAN, torch.nn.Sigmoid),
    (ConditionalVanillaGAN, torch.nn.Sigmoid),
    (ConditionalWassersteinGAN, torch.nn.Identity),
    (ConditionalWassersteinGANGP, torch.nn.Identity),
    (ConditionalPix2Pix, torch.nn.Sigmoid)
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

            )
            self.feature_part = torch.nn.Linear(16, np.prod(out_dim))
            self.feature_reshape = LayerReshape(out_dim)
            self.output = last_layer()

        def forward(self, x):
            x = self.hidden_part(x)
            x = self.feature_part(x)
            x = self.feature_reshape(x)
            return self.output(x)

    return MyNetwork(in_dim, last_layer, out_dim)


@pytest.mark.parametrize("gan, last_layer", networks)
def test_init(gan, last_layer):
    z_dim = 10
    y_dim = 5

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=16)
    adv = generate_net(in_dim=21, last_layer=last_layer, out_dim=1)

    testgan = gan(generator=gen, adversary=adv, x_dim=16, z_dim=z_dim, y_dim=y_dim, folder=None)
    with pytest.raises(TypeError):
        gan(generator=gen, adversary=adv, x_dim=17, z_dim=z_dim, y_dim=y_dim, folder=None)
    with pytest.raises(TypeError):
        gan(generator=gen, adversary=adv, x_dim=16, z_dim=11, y_dim=y_dim, folder=None)

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=17)
    adv = generate_net(in_dim=21, last_layer=last_layer, out_dim=1)
    with pytest.raises(AssertionError):
        gan(generator=gen, adversary=adv, x_dim=16, z_dim=z_dim, y_dim=y_dim, folder=None)


@pytest.mark.parametrize("gan, last_layer", networks)
def test_fit_vector(gan, last_layer):
    z_dim = 10
    y_dim = 5
    X_train = np.zeros(shape=[100, 16])
    y_train = np.zeros(shape=[100, y_dim])
    X_test = np.zeros(shape=[100, 16])
    y_test = np.zeros(shape=[100, y_dim])

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=16)
    adv = generate_net(in_dim=21, last_layer=last_layer, out_dim=1)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    testgan = gan(generator=gen, adversary=adv, x_dim=16, z_dim=z_dim, y_dim=y_dim, folder=None)
    testgan.fit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, **fit_kwargs
    )
    assert hasattr(testgan, "logged_losses")


@pytest.mark.parametrize("gan, last_layer", networks)
def test_fit_vector_feature_loss(gan, last_layer):
    z_dim = 10
    y_dim = 5

    X_train = np.zeros(shape=[100, 16])
    y_train = np.zeros(shape=[100, y_dim])
    X_test = np.zeros(shape=[100, 16])
    y_test = np.zeros(shape=[100, y_dim])

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=16)
    adv = generate_net(in_dim=21, last_layer=last_layer, out_dim=1)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    testgan = gan(generator=gen, adversary=adv, x_dim=16, z_dim=z_dim, y_dim=y_dim, folder=None, feature_layer=adv.hidden_part)
    testgan.fit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, **fit_kwargs
    )
    assert hasattr(testgan, "logged_losses")


@pytest.mark.parametrize("gan, last_layer", networks)
def test_fit_error_vector(gan, last_layer):
    z_dim = 10
    y_dim = 5
    X_train = np.zeros(shape=[100, 16])
    X_train_wrong_shape1 = np.zeros(shape=[100])
    X_train_wrong_shape2 = np.zeros(shape=[100, 17])
    X_train_wrong_shape3 = np.zeros(shape=[100, 17, 14])
    X_train_wrong_shape4 = np.zeros(shape=[100, 17, 10, 10])
    X_test_wrong_shape = np.zeros(shape=[100, 17])

    y_train = np.zeros(shape=[100, y_dim])
    y_test = np.zeros(shape=[100, y_dim])

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=16)
    adv = generate_net(in_dim=21, last_layer=last_layer, out_dim=1)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    testgan = gan(generator=gen, adversary=adv, x_dim=16, z_dim=z_dim, y_dim=y_dim, folder=None)

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape1, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape2, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape3, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape4, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train, y_train=y_train, X_test=X_test_wrong_shape, y_test=y_test, **fit_kwargs
        )


@pytest.mark.parametrize("gan, last_layer", networks)
def test_fit_image(gan, last_layer):
    z_dim = 10
    y_dim = 5
    x_dim = [3, 16, 16]

    X_train = np.zeros(shape=[100, *x_dim])
    y_train = np.zeros(shape=[100, y_dim])
    X_test = np.zeros(shape=[100, *x_dim])
    y_test = np.zeros(shape=[100, y_dim])

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=x_dim)
    adv = generate_net(in_dim=get_input_dim(x_dim, y_dim), last_layer=last_layer, out_dim=1)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    testgan = gan(generator=gen, adversary=adv, x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder=None)
    testgan.fit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, **fit_kwargs
    )
    assert hasattr(testgan, "logged_losses")


@pytest.mark.parametrize("gan, last_layer", networks)
def test_fit_image_feature_loss(gan, last_layer):
    z_dim = 10
    y_dim = 5
    x_dim = [3, 16, 16]

    X_train = np.zeros(shape=[100, *x_dim])
    y_train = np.zeros(shape=[100, y_dim])
    X_test = np.zeros(shape=[100, *x_dim])
    y_test = np.zeros(shape=[100, y_dim])

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=x_dim)
    adv = generate_net(in_dim=get_input_dim(x_dim, y_dim), last_layer=last_layer, out_dim=1)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    testgan = gan(generator=gen, adversary=adv, x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder=None, feature_layer=adv.hidden_part)
    testgan.fit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, **fit_kwargs
    )
    assert hasattr(testgan, "logged_losses")


@pytest.mark.parametrize("gan, last_layer", networks)
def test_fit_error_image(gan, last_layer):
    z_dim = 10
    y_dim = 5
    x_dim = [3, 16, 16]
    X_train = np.zeros(shape=[100, *x_dim])
    X_train_wrong_shape1 = np.zeros(shape=[100])
    X_train_wrong_shape2 = np.zeros(shape=[100, 17])
    X_train_wrong_shape3 = np.zeros(shape=[100, 17, 14])
    X_train_wrong_shape4 = np.zeros(shape=[100, 4, 16, 16])
    X_test_wrong_shape = np.zeros(shape=[100, 3, 17, 17])

    y_train = np.zeros(shape=[100, y_dim])
    y_test = np.zeros(shape=[100, y_dim])

    gen = generate_net(in_dim=15, last_layer=torch.nn.Sigmoid, out_dim=x_dim)
    adv = generate_net(in_dim=get_input_dim(x_dim, y_dim), last_layer=last_layer, out_dim=1)
    fit_kwargs = {
        "epochs": 1, "batch_size": 4, "steps": None, "print_every": None, "save_model_every": None,
        "save_images_every": None, "save_losses_every": "1e", "enable_tensorboard": False
    }
    testgan = gan(generator=gen, adversary=adv, x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder=None)

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape1, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape2, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape3, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train_wrong_shape4, y_train=y_train, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan.fit(
            X_train=X_train, y_train=y_train, X_test=X_test_wrong_shape, y_test=y_test, **fit_kwargs
        )

    with pytest.raises(AssertionError):
        testgan = gan(generator=gen, adversary=adv, x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder=None, feature_layer="hidden_part")


@pytest.mark.parametrize("gan, optim",
    [
        (ConditionalKLGAN, torch.optim.Adam),
        (ConditionalLSGAN, torch.optim.Adam),
        (ConditionalVanillaGAN, torch.optim.Adam),
        (ConditionalWassersteinGAN, torch.optim.RMSprop),
        (ConditionalWassersteinGANGP, torch.optim.RMSprop),
        (ConditionalPix2Pix, torch.optim.Adam)
    ]
)
def test_default_optimizers(gan, optim):
    assert gan._default_optimizer(gan) == optim
