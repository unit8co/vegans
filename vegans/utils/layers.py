import torch
from torch.nn import Module

class LayerPrintSize(Module):
    """ Prints the size of a layer without performing any operation.

    Mainly used for debugging to find the layer shape at a certain depth of the network.
    """
    def __init__(self):
        super(LayerPrintSize, self).__init__()

    def forward(self, x):
        print("\n")
        print(x.shape)
        print("\n")
        return x


class LayerReshape(Module):
    """ Reshape a tensor.

    Might be used in a densely connected network in the last layer to produce an image output.
    """
    def __init__(self, shape):
        super(LayerReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        x = torch.reshape(input=x, shape=(-1, *self.shape))
        return x

    def __str__(self):
        return "LayerReshape(shape="+str(self.shape)+")"
