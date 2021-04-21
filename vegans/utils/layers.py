import torch
from torch.nn import Module

class LayerPrintSize(Module):
    """ Prints the size of a layer without performing any operation.

    Mainly used for debugging to find the layer shape at a certain depth of the network.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("\n")
        print("Layer shape:", x.shape)
        return x


class LayerReshape(Module):
    """ Reshape a tensor.

    Might be used in a densely connected network in the last layer to produce an image output.
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        x = torch.reshape(input=x, shape=(-1, *self.shape))
        return x

    def __str__(self):
        return "LayerReshape(shape="+str(self.shape)+")"

    def __repr__(self):
        return "LayerReshape(shape="+str(self.shape)+")"


class LayerInception(Module):
    """ Implementation of the inception layer architecture.

    Uses a network in network (NIN) architecture to make networks wider
    and deeper.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0, (
            "`LayerInception` out_channels must be divisible by four. Given: {}.".format(out_channels)
        )
        out_channels = out_channels // 4
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.one_by_one1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

        self.one_by_one2 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )
        self.three_by_three = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )

        self.one_by_one3 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )
        self.five_by_five = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2
        )

        self.max_pooling = torch.nn.MaxPool2d(
            kernel_size=5, stride=1, padding=2
        )
        self.one_by_one4 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )


    def forward(self, x):
        layer1 = self.one_by_one1(x)
        layer2 = self.three_by_three(self.one_by_one2(x))
        layer3 = self.five_by_five(self.one_by_one3(x))
        layer4 = self.one_by_one4(self.max_pooling(x))
        x = torch.cat((layer1, layer2, layer3, layer4), axis=1)
        return x

    def __str__(self):
        return "LayerInception(in_channels={}, out_channels={})".format(self.in_channels, self.out_channels)

    def __repr__(self):
        return "LayerInception(in_channels={}, out_channels={})".format(self.in_channels, self.out_channels)


class LayerResidualBlock(Module):
    """ Implementation of the inception layer architecture.

    Uses a network in network (NIN) architecture to make networks wider
    and deeper.
    """
    def __init__(self, in_channels, out_channels, skip_layers, kernel_size):
        super().__init__()
        assert isinstance(out_channels, int) and out_channels > in_channels, (
            "`out_channels` must be a larger integer than `in_channels` due to concatenation. " +
            "in_channels: {}. out_channels: {}.".format(in_channels, out_channels)
        )
        assert isinstance(skip_layers, int) and skip_layers > 0, (
            "`skip_layers` must be a positive integer. Given: {}.".format(skip_layers)
        )
        assert isinstance(kernel_size, int) and kernel_size % 2 == 1, (
            "`kernel_size` must be an odd integer. Given: {}.".format(kernel_size)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = self.out_channels - self.in_channels
        self.skip_layers = skip_layers
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.skip0 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=self.skip_channels, kernel_size=kernel_size, stride=1, padding=self.padding
        )

        for i in range(1, self.skip_layers+1):
            setattr(self, "skip{}".format(i), torch.nn.Conv2d(
                    in_channels=self.skip_channels, out_channels=self.skip_channels, kernel_size=kernel_size, stride=1, padding=self.padding
                )
            )

    def forward(self, x):
        out = x
        for i in range(self.skip_layers+1):
            layer = getattr(self, "skip{}".format(i))
            out = layer(out)
        x = torch.cat((out, x), axis=1)
        return x

    def __str__(self):
        return (
            "LayerResidualBlock(in_channels={}, out_channels={}, skip_layers={}, kernel_size={})"
            .format(self.in_channels, self.out_channels, self.skip_layers, self.kernel_size)
        )

    def __repr__(self):
        return (
            "LayerResidualBlock(in_channels={}, out_channels={}, skip_layers={}, kernel_size={})"
            .format(self.in_channels, self.out_channels, self.skip_layers, self.kernel_size)
        )
