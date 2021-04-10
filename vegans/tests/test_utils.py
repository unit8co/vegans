import pytest
import vegans.utils.utils as utils

def test_Dataset():
    X = list(range(100))
    data = utils.DataSet(X)
    assert len(data) == len(X)


def test_load_mnist():
    datapath = "./"