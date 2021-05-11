import torch
import pytest
import shutil

import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

from vegans.GAN import ConditionalVanillaGAN


# def teardown_module(module):
#     print('******TEARDOWN******')
#     default_directory = loading.MNISTLoader()._root
#     shutil.rmtree(default_directory)

def test_MNISTLoader():
    loader = loading.MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load()
    assert X_train.shape == (60000, 1, 32, 32)
    assert y_train.shape == (60000, 10)
    assert X_test.shape == (10000, 1, 32, 32)
    assert y_test.shape == (10000, 10)
    generator = loader.load_generator()
    adversary = loader.load_adversary()
    gan_model = ConditionalVanillaGAN(
        generator=generator, adversary=adversary, x_dim=(1, 32, 32), z_dim=32, y_dim=10, folder=None
    )

def test_FashionMNISTLoader():
    loader = loading.FashionMNISTLoader()
    X_train, y_train, X_test, y_test = loader.load()
    assert X_train.shape == (60000, 1, 32, 32)
    assert y_train.shape == (60000, 10)
    assert X_test.shape == (10000, 1, 32, 32)
    assert y_test.shape == (10000, 10)

def test_Cifar10Loader():
    loader = loading.CIFAR10Loader()
    X_train, y_train, X_test, y_test = loader.load()
    assert X_train.shape == (50000, 3, 32, 32)
    assert y_train.shape == (50000, 10)
    assert X_test.shape == (10000, 3, 32, 32)
    assert y_test.shape == (10000, 10)
    generator = loader.load_generator()
    adversary = loader.load_adversary()
    gan_model = ConditionalVanillaGAN(
        generator=generator, adversary=adversary, x_dim=(3, 32, 32), z_dim=64, y_dim=10, folder=None
    )

def test_Cifar100Loader():
    loader = loading.CIFAR100Loader()
    X_train, y_train, X_test, y_test = loader.load()
    assert X_train.shape == (50000, 3, 32, 32)
    assert y_train.shape == (50000, 100)
    assert X_test.shape == (10000, 3, 32, 32)
    assert y_test.shape == (10000, 100)

def test_CelebALoader():
    loader = loading.CelebALoader()
    train_loader = loader.load()
    X_train, y_train = iter(train_loader).__next__()
    X_train, y_train = X_train.numpy(), y_train.numpy()
    assert X_train.shape == (32, 3, 218, 178)
    assert y_train.shape == (32, 40)

