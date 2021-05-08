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