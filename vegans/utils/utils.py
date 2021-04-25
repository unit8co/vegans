import torch
import pickle

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from vegans.utils.layers import LayerReshape

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

class WassersteinLoss():
    def __call__(self, input, target):
        """ Computes the Wasserstein loss / divergence.

        Also known as earthmover distance.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor. Output of a critic.
        target : torch.Tensor
            Label, either 1 or -1. Zeros are translated to -1.

        Returns
        -------
        torch.Tensor
            Wasserstein divergence
        """
        assert torch.unique(target).shape[0] <= 2, "Only two different values for target allowed."
        target[target==0] = -1

        return torch.mean(target*input)

class NormalNegativeLogLikelihood():
    def __call__(self, x, mu, variance, eps=1e-6):
        negative_log_likelihood = 1/(2*variance + eps)*(x-mu)**2 + 0.5*torch.log(variance + eps)
        negative_log_likelihood = negative_log_likelihood.sum(axis=1).mean()
        return negative_log_likelihood

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
        return torch.cat((tensor1, tensor2), axis=1).float()
    elif (len(tensor1.shape) == 2) and (len(tensor2.shape) == 2):
        return torch.cat((tensor1, tensor2), axis=1).float()
    elif (len(tensor1.shape) == 4) and (len(tensor2.shape) == 2):
        y_dim = tensor2.shape[1]
        tensor2 = torch.reshape(tensor2, shape=(batch_size, y_dim, 1, 1))
        tensor2 = torch.tile(tensor2, dims=(1, 1, *tensor1.shape[2:]))
    elif (len(tensor1.shape) == 2) and (len(tensor2.shape) == 4):
        y_dim = tensor1.shape[1]
        tensor1 = torch.reshape(tensor1, shape=(batch_size, y_dim, 1, 1))
        tensor1 = torch.tile(tensor1, dims=(1, 1, *tensor2.shape[2:]))
    elif (len(tensor1.shape) == 4) and (len(tensor2.shape) == 4):
        return torch.cat((tensor1, tensor2), axis=1).float()
    else:
        raise AssertionError("tensor1 and tensor2 must have 2 or 4 dimensions. Given: {} and {}.".format(tensor1.shape, tensor2.shape))
    return torch.cat((tensor1, tensor2), axis=1).float()

def get_input_dim(dim1, dim2):
    """ Get the number of input dimension from two inputs.

    Tensors often need to be concatenated in different ways, especially for conditional algorithms
    leveraging label information. This function returns the output dimensions of a tensor after the concatenation of
    two 2D tensors (two vectors), two 4D tensors (two images) or one 2D tensor with another 4D Tensor (vector with image).
    For both tensors the first dimension will be number of samples which is not considered in this function.
    Therefore `dim1` and `dim2` are both either 1D or 3D Tensors indicating the vector or
    image dimensions (nr_channles, height, width).
    In a usual use case `dim1` is either the latent z dimension (often a vector) or a sample from the sample space
    (might be an image). `dim2` often represents the conditional y dimension that is concatenated with the noise
    or a sample vefore passing it to a neural network.

    This function ca be used to get the input dimension for the generator, adversary, encoder or decoder in a
    conditional use case.

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
    return tuple(out_dim)

def plot_losses(losses, show=True, share=False):
    """
    Plots losses for generator and discriminator on a common plot.

    Parameters
    ----------
    losses : dict
        Dictionary containing the losses for some networks. The structure of the dictionary is:
        ```
        {
            mode1: {loss_type1_1: losses1_1, loss_type1_2: losses1_2, ...},
            mode2: {loss_type2_1: losses2_1, loss_type2_2: losses2_2, ...},
            ...
        }
        ```
        where `mode` is probably one of "Train" or "Test", loss_type might be "Generator", "Adversary", "Encoder", ...
        and losses are lists of loss values collected during training.
    show : bool, optional
        If True, `plt.show` is called to visualise the images directly.
    share : bool, optional
        If true, axis ticks are shared between plots.

    Returns
    -------
    plt.figure, plt.axis
        Created figure and axis objects.
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
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9))
        axs = np.ravel(axs)
        for mode, loss_dict in losses.items():
            for ax, (loss_type, loss) in zip(axs, loss_dict.items()):
                ax.plot(loss, lw=2, label=mode)
                ax.set_xlabel('Iterations')
                ax.set_title(loss_type)
                ax.set_facecolor("#ecffe7")
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
        Must be of shape [nr_samples, height, width] or [nr_samples, height, width, 3].
    labels : np.array, optional
        Array of labels used in the title.
    show : bool, optional
        If True, `plt.show` is called to visualise the images directly.
    n : None, optional
        Number of images to be drawn, maximum is 36.

    Returns
    -------
    plt.figure, plt.axis
        Created figure and axis objects.
    """
    if len(images.shape)==4 and images.shape[1] == 3:
        images = invert_channel_order(images=images)
    elif len(images.shape)==4 and images.shape[1] == 1:
        images = images.reshape((-1, images.shape[2], images.shape[3]))
    if n is None:
        n = images.shape[0]
    if n > 36:
        n = 36
    nrows = int(np.sqrt(n))
    ncols = n // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))
    axs = np.ravel(axs)

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis("off")
        if labels is not None:
            ax.set_title("Label: {}".format(labels[i]))

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs

def create_gif(source_path, target_path=None):
    """Create a GIF from images contained on the source path.

    Parameters
    ----------
    source_path : string
        Path pointing to the source directory with .png files.
    target_path : string, optional
        Name of the created GIF.
    """
    import os
    import imageio
    source_path = source_path+"/" if not source_path.endswith("/") else source_path
    images = []
    for file_name in sorted(os.listdir(source_path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(source_path, file_name)
            images.append(imageio.imread(file_path))

    if target_path is None:
        target_path = source_path+"movie.gif"
    imageio.mimsave(target_path, images)


def invert_channel_order(images):
    assert len(images.shape) == 4, "`images` must be of shape [batch_size, nr_channels, height, width]. Given: {}.".format(images.shape)
    assert images.shape[1] == 3 or images.shape[3] == 3, (
        "`images` must have 3 colour channels at second or fourth shape position. Given: {}.".format(images.shape)
    )
    inverted_images = []

    if images.shape[1] == 3:
        image_y = images.shape[2]
        image_x = images.shape[3]
        for i, image in enumerate(images):
            red_channel = image[0].reshape(image_y, image_x)
            green_channel = image[1].reshape(image_y, image_x)
            blue_channel = image[2].reshape(image_y, image_x)
            image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
            inverted_images.append(image)
    elif images.shape[3] == 3:
        image_y = images.shape[1]
        image_x = images.shape[2]
        for i, image in enumerate(images):
            red_channel = image[:, :, 0].reshape(image_y, image_x)
            green_channel = image[:, :, 1].reshape(image_y, image_x)
            blue_channel = image[:, :, 2].reshape(image_y, image_x)
            image = np.stack((red_channel, green_channel, blue_channel), axis=0)
            inverted_images.append(image)
    return np.array(inverted_images)