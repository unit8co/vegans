import torch
import pytest

import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading


def test_MNISTLoader():
    loader = loading.MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load()
    assert X_train.shape == (60000, 1, 32, 32)
    assert y_train.shape == (60000, 10)
    assert X_test.shape == (10000, 1, 32, 32)
    assert y_test.shape == (10000, 10)

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

