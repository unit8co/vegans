import torch
import pytest

import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

def test_Dataset():
    X = list(range(100))
    data = utils.DataSet(X)
    assert len(data) == len(X)


def test_load_mnist():
    datapath = "./data/"
    X_train, y_train, X_test, y_test = loading.load_data(datapath, which="MNIST", download=True)
    assert X_train.shape == (60000, 32, 32)
    assert np.max(X_train) == 1
    assert X_test.shape == (10000, 32, 32)
    assert np.max(X_test) == 1

def test_wasserstein_loss():
    labels = torch.from_numpy(np.array([1, 1, 0, 0, 1, 0])).float()
    predictions = torch.from_numpy(np.array([5, 3, -2, 3, 8, -2])).float()

    w_loss = utils.wasserstein_loss(input=predictions, target=labels).cpu().numpy()
    labels[labels==0] = -1
    result = torch.mean(predictions*labels).cpu().numpy()
    assert w_loss == result

    labels = torch.from_numpy(np.array([1, 1, 2, 0, 1, 0])).float()
    with pytest.raises(AssertionError) as e_info:
        utils.wasserstein_loss(input=predictions, target=labels)

def test_concatenate():
    tensor1 = torch.randn(20, 5, requires_grad=False, device="cpu")
    tensor2 = torch.randn(20, 10, requires_grad=False, device="cpu")
    result = utils.concatenate(tensor1, tensor2)
    assert result.numpy().shape == (20, 15)

    tensor1 = torch.randn(20, 5, 4, 4, requires_grad=False, device="cpu")
    tensor2 = torch.randn(20, 10, requires_grad=False, device="cpu")
    result = utils.concatenate(tensor1, tensor2)
    assert result.numpy().shape == (20, 15, 4, 4)

    tensor1 = torch.randn(20, 5, requires_grad=False, device="cpu")
    tensor2 = torch.randn(20, 10, 8, 7, requires_grad=False, device="cpu")
    result = utils.concatenate(tensor1, tensor2)
    assert result.numpy().shape == (20, 15, 8, 7)

    tensor1 = torch.randn(20, 5, 6, 6, requires_grad=False, device="cpu")
    tensor2 = torch.randn(20, 1, 6, 6, requires_grad=False, device="cpu")
    result = utils.concatenate(tensor1, tensor2)
    assert result.numpy().shape == (20, 6, 6, 6)

    with pytest.raises(AssertionError):
        tensor1 = torch.randn(20, 5, 5, 5, requires_grad=False, device="cpu")
        tensor1 = torch.randn(22, 5, 5, 5, requires_grad=False, device="cpu")
        utils.concatenate(tensor1, tensor2)

    with pytest.raises(AssertionError):
        tensor1 = torch.randn(20, 5, 5, requires_grad=False, device="cpu")
        utils.concatenate(tensor1, tensor2)

    with pytest.raises(AssertionError):
        tensor1 = torch.randn(20, requires_grad=False, device="cpu")
        utils.concatenate(tensor1, tensor2)

    with pytest.raises(AssertionError):
        tensor1 = torch.randn(20, 5, 6, 5, requires_grad=False, device="cpu")
        tensor1 = torch.randn(22, 5, 5, 5, requires_grad=False, device="cpu")
        utils.concatenate(tensor1, tensor2)

def test_get_input_dim():
    dim1 = 12
    dim2 = 34
    assert utils.get_input_dim(dim1, dim2) == tuple([46])

    dim1 = [12]
    dim2 = [34]
    assert utils.get_input_dim(dim1, dim2) == tuple([46])

    dim1 = [1, 3, 4]
    dim2 = [5]
    assert utils.get_input_dim(dim1, dim2) == tuple([6, 3, 4])

    dim1 = 5
    dim2 = [1, 3, 4]
    assert utils.get_input_dim(dim1, dim2) == tuple([6, 3, 4])

    dim1 = [3, 4, 6]
    dim2 = [7, 4, 6]
    assert utils.get_input_dim(dim1, dim2) == tuple([10, 4, 6])

    with pytest.raises(AssertionError):
        dim1 = [3, 4]
        utils.get_input_dim(dim1, dim2)

    with pytest.raises(AssertionError):
        dim1 = [20, 1, 4, 4]
        utils.get_input_dim(dim1, dim2)

    with pytest.raises(AssertionError):
        dim1 = [3, 4, 4]
        dim2 = [3, 4, 5]
        utils.get_input_dim(dim1, dim2)

