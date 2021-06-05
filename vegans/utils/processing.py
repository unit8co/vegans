import torch
import numpy as np
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