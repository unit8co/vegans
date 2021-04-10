import torch
import pickle

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]


def load_mnist(datapath, normalize=True, pad=None, return_datasets=False):
    datapath = datapath if datapath.endswith("/") else datapath+"/"
    with open(datapath+"train_images.pickle", "rb") as f:
        X_train, y_train = pickle.load(f)
    with open(datapath+"test_images.pickle", "rb") as f:
        X_test, y_test = pickle.load(f)

    if normalize:
        max_number = X_train.max()
        X_train = X_train / max_number
        X_test = X_test / max_number

    if pad:
        X_train = np.pad(X_train, [(0, 0), (pad, pad), (pad, pad)], mode='constant')
        X_test = np.pad(X_test, [(0, 0), (pad, pad), (pad, pad)], mode='constant')

    if return_datasets:
        train = DataSet(X_train, y_train)
        test = DataSet(X_test, y_test)
        return(train, test)
    return X_train, y_train, X_test, y_test


def wasserstein_loss(input, target):
    assert torch.unique(target).shape[0] == 2, "Only two different values for target allowed."
    target[target==0] = -1

    return torch.mean(target*input)


def concatenate(tensor1, tensor2):
    """ Concatenates two 2D or 4D tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        2D or 4D tensor.
    tensor2 : torch.Tensor
        2D or 4D tensor.

    Returns
    -------
    torch.Tensor
        Cncatenation of tensor1 and tensor2.

    Raises
    ------
    NotImplementedError
        If tensors do not have 2 or 4 dimensions.
    """
    assert tensor1.shape[0] == tensor2.shape[0], (
        "Tensors to concatenate must have same dim 0. Tensor1: {}. Tensor2: {}.".format(tensor1.shape[0], tensor2.shape[0])
    )
    batch_size = tensor1.shape[0]
    if tensor1.shape == tensor2.shape:
        return torch.cat((tensor1, tensor2), axis=1)
    elif (len(tensor1.shape) == 2) and (len(tensor2.shape) == 2):
        return torch.cat((tensor1, tensor2), axis=1)
    elif (len(tensor1.shape) == 4) and (len(tensor2.shape) == 2):
        y_dim = tensor2.shape[1]
        tensor2 = torch.reshape(tensor2, shape=(batch_size, y_dim, 1, 1))
        tensor2 = torch.tile(tensor2, dims=(1, 1, *tensor1.shape[2:]))
    elif (len(tensor1.shape) == 2) and (len(tensor2.shape) == 4):
        y_dim = tensor1.shape[1]
        tensor1 = torch.reshape(tensor1, shape=(batch_size, y_dim, 1, 1))
        tensor1 = torch.tile(tensor1, dims=(1, 1, *tensor2.shape[2:]))
    elif (len(tensor1.shape) == 4) and (len(tensor2.shape) == 4):
        return torch.cat((tensor1, tensor2), axis=1)
    else:
        raise AssertionError("tensor1 and tensor2 must have 2 or 4 dimensions. Given: {} and {}.".format(tensor1.shape, tensor2.shape))
    return torch.cat((tensor1, tensor2), axis=1)

def get_input_dim(dim1, dim2):
    """ Get the number of input dimension from two inputs.

    Tensors often need to be concatenated in different ways. This library
    supports the concatenation of a 2D tensor with a 4D Tensor. For both
    tensors the first dimension will be number of samples which is not
    considered in this function. Therefore pass a 1D or 3D Tensor indicating
    the vector or image dimensions (nr_channles, height, width). Calculates the
    output dim when concatenating two images, an image with a vector or a vector
    with an image.

    Parameters
    ----------
    dim1 : int, iterable
        Dimension of input 1.
    dim2 : int, iterable
        Dimension of input 2.

    Returns
    -------
    list
        Output dimension after concatenation.
    """
    dim1 = [dim1] if isinstance(dim1, int) else dim1
    dim2 = [dim2] if isinstance(dim2, int) else dim2
    if len(dim1)==1 and len(dim2)==1:
        out_dim = [dim1[0] + dim2[0]]
    elif len(dim1)==3 and len(dim2)==1:
        out_dim = [dim1[0]+dim2[0], *dim1[1:]]
    elif len(dim1)==1 and len(dim2)==3:
        out_dim = [dim2[0]+dim1[0], *dim2[1:]]
    elif len(dim1)==3 and len(dim2)==3:
        assert (dim1[1] == dim2[1]) and (dim1[2] == dim2[2]), (
            "If both dim1 and dim2 are arrays, must have same shape. dim1: {}. dim2: {}.".format(dim1, dim2)
        )
        out_dim = [dim1[0]+dim2[0], *dim1[1:]]
    else:
        raise AssertionError("dim1 and dim2 must have length one or three. Given: {} and {}.".format(dim1, dim2))
    return out_dim

def plot_losses(losses, show=True, share=False):
    """
    Plots losses for generator and discriminator on a common plot.

    Parameters
    ----------
    losses : dict
        Dictionary containing the losses for some networks.
    """
    if share:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for mode, loss_dict in losses.items():
            for loss_type, loss in loss_dict.items():
                ax.plot(loss, lw=2, label=mode+loss_type)
        ax.set_xlabel('Iterations')
        ax.legend()
    else:
        n = len(losses["Train"])
        nrows = int(np.sqrt(n))
        ncols = n // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        axs = np.ravel(axs)
        for mode, loss_dict in losses.items():
            for ax, (loss_type, loss) in zip(axs, loss_dict.items()):
                ax.plot(loss, lw=2, label=mode)
                ax.set_xlabel('Iterations')
                ax.set_title(loss_type)
                ax.legend()
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_images(images, labels=None, show=True, n=None):
    """ Plot a number of input images with optional label

    Parameters
    ----------
    images : np.array
        Must be of shape [nr_samples, height, width].
    labels : np.array, optional
        Array of labels used in the title.
    show : bool, optional
        If True, `plt.show` is called to visualise the images directly.
    n : None, optional
        Number of images to be drawn, maximum is 36.

    Returns
    -------
    TYPE
        Description
    """
    if n is None:
        n = images.shape[0]
    if n > 36:
        n = 36
    nrows = int(np.sqrt(n))
    ncols = n // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    axs = np.ravel(axs)

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis("off")
        if labels is not None:
            ax.set_title("Label: {}".format(labels[i]))
    if show:
        plt.show()
    return fig, axs