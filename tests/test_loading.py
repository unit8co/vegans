import torch
import pytest
import shutil

import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

from vegans.GAN import ConditionalVanillaGAN, ConditionalBicycleGAN, ConditionalVanillaVAE


# def teardown_module(module):
#     print('******TEARDOWN******')
#     default_directory = loading.MNISTLoader()._root
#     shutil.rmtree(default_directory)

# def test_MNISTLoader():
#     loader = loading.MNISTLoader()
#     X_train, y_train, X_test, y_test = loader.load()
#     assert X_train.shape == (60000, 1, 32, 32)
#     assert y_train.shape == (60000, 10)
#     assert X_test.shape == (10000, 1, 32, 32)
#     assert y_test.shape == (10000, 10)
#     generator = loader.load_generator()
#     adversary = loader.load_adversary()
#     gan_model = ConditionalVanillaGAN(
#         generator=generator, adversary=adversary, x_dim=(1, 32, 32), z_dim=32, y_dim=10, folder=None
#     )

# def test_FashionMNISTLoader():
#     loader = loading.FashionMNISTLoader()
#     X_train, y_train, X_test, y_test = loader.load()
#     assert X_train.shape == (60000, 1, 32, 32)
#     assert y_train.shape == (60000, 10)
#     assert X_test.shape == (10000, 1, 32, 32)
#     assert y_test.shape == (10000, 10)

# def test_Cifar10Loader():
#     loader = loading.CIFAR10Loader()
#     X_train, y_train, X_test, y_test = loader.load()
#     assert X_train.shape == (50000, 3, 32, 32)
#     assert y_train.shape == (50000, 10)
#     assert X_test.shape == (10000, 3, 32, 32)
#     assert y_test.shape == (10000, 10)
#     generator = loader.load_generator()
#     adversary = loader.load_adversary()
#     gan_model = ConditionalVanillaGAN(
#         generator=generator, adversary=adversary, x_dim=(3, 32, 32), z_dim=64, y_dim=10, folder=None
#     )

# def test_Cifar100Loader():
#     loader = loading.CIFAR100Loader()
#     X_train, y_train, X_test, y_test = loader.load()
#     assert X_train.shape == (50000, 3, 32, 32)
#     assert y_train.shape == (50000, 100)
#     assert X_test.shape == (10000, 3, 32, 32)
#     assert y_test.shape == (10000, 100)

def test_CelebALoader():
    batch_size, max_loaded_images, crop_size = 32, 200, 150
    for output_shape in [64, 128, 256]:
        loader = loading.CelebALoader(
            batch_size=batch_size, max_loaded_images=max_loaded_images, crop_size=crop_size, output_shape=output_shape
        )
        train_loader = loader.load()
        X_train, y_train = iter(train_loader).__next__()
        X_train, y_train = X_train.numpy(), y_train.numpy()
        assert X_train.shape == (batch_size, 3, output_shape, output_shape)
        assert y_train.shape == (batch_size, 40)

        generator = loader.load_generator()
        adversary = loader.load_adversary()
        encoder = loader.load_encoder()
        decoder = loader.load_decoder()
        gan_model = ConditionalVanillaGAN(
            generator=generator, adversary=adversary,
            x_dim=(3, output_shape, output_shape), z_dim=(16, 4, 4), y_dim=40,
            folder=None
        )
        gan_model = ConditionalBicycleGAN(
            generator=generator, adversary=adversary, encoder=encoder,
            x_dim=(3, output_shape, output_shape), z_dim=(16, 4, 4), y_dim=40,
            folder=None
        )
        gan_model = ConditionalVanillaVAE(
            encoder=encoder, decoder=decoder,
            x_dim=(3, output_shape, output_shape), z_dim=(16, 4, 4), y_dim=40,
            folder=None
        )

    with pytest.raises(AssertionError): # `z_dim[1]` must be divisible by 2
        generator = loader.load_generator(z_dim=(16, 5, 4))

    with pytest.raises(AssertionError): # `x_dim[1]` must be divisible by 2
        generator = loader.load_generator(x_dim=(16, 5, 4))

    with pytest.raises(AssertionError): # `x_dim[1]` must be divisible by `z_dim[1]`
        generator = loader.load_generator(x_dim=(16, 16, 16), z_dim=(16, 14, 14))

    with pytest.raises(AssertionError): # `x_dim[1]/z_dim[1]` must be divisible by 2
        generator = loader.load_generator(x_dim=(16, 12, 12), z_dim=(16, 4, 4))

    with pytest.raises(AssertionError): # `z_dim[1]` must be equal to `z_dim[2]`
        generator = loader.load_generator(z_dim=(16, 2, 4))