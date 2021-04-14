import re
import json
import torch

import numpy as np

from torch import nn
from torchsummary import summary
from torch.nn import Module, Sequential


class NeuralNetwork(Module):
    """ Basic abstraction for single networks.

    These networks form the building blocks for the generative adversarial networks.
    Mainly responsible for consistency checks.
    """
    def __init__(self, network, name, input_size, device, ngpu):
        super(NeuralNetwork, self).__init__()
        self.name = name
        self.input_size = input_size
        self.device = device
        self.ngpu = ngpu
        if isinstance(input_size, int):
            self.input_size = tuple([input_size])
        elif isinstance(input_size, list):
            self.input_size = tuple(input_size)

        assert isinstance(network, torch.nn.Module), "network must inherit from nn.Module."
        try:
            type(network[-1])
            self.input_type = "Sequential"
        except TypeError:
            self.input_type = "Object"
        self.network = network.to(self.device)
        self._validate_input()

        if self.device=="cuda" and self.ngpu is not None:
            if self.ngpu > 1:
                self.network = torch.nn.DataParallel(self.network)

        self.output_size = self._get_output_shape()[1:]

    def forward(self, x):
        output = self.network(x)
        return output

    def _validate_input(self):
        iterative_layers = self._get_iterative_layers(self.network, self.input_type)

        for layer in iterative_layers:
            if "in_features" in layer.__dict__:
                first_input = layer.__dict__["in_features"]
                break
            elif "in_channels" in layer.__dict__:
                first_input = layer.__dict__["in_channels"]
                break
            elif "num_features" in layer.__dict__:
                first_input = layer.__dict__["num_features"]
                break

        if np.prod([first_input]) == np.prod(self.input_size):
            pass
        elif (len(self.input_size) > 1) & (self.input_size[0] == first_input):
            pass
        else:
            raise TypeError(
                "\n\tInput mismatch for {}:\n".format(self.name) +
                "\t\tFirst input layer 'in_features' or 'in_channels': {}. self.input_size: {}.\n".format(
                    first_input, self.input_size) +
                "\t\tIf you are trying to use a conditional model please make sure you adjusted the input size\n" +
                "\t\tof the first layer in this architecture for the label vector / image.\n"
                "\t\tIn this case, use vegans.utils.utils.get_input_dim(in_dim, y_dim) and adjust this architecture's\n" +
                "\t\tfirst layer input accordingly. See the conditional examples on github for help."
            )
        return True

    @staticmethod
    def _get_iterative_layers(network, input_type):
        if input_type == "Sequential":
            return network
        elif input_type == "Object":
            iterative_net = []
            for _, layers in network.__dict__["_modules"].items():
                try:
                    for layer in layers:
                        iterative_net.append(layer)
                except TypeError:
                    iterative_net.append(layers)
            return iterative_net
        else:
            raise NotImplemented("Network must be Sequential or Object.")

    def _get_output_shape(self):
        sample_input = torch.rand([2, *self.input_size]).to(self.device)
        return self.network(sample_input).data.cpu().numpy().shape


    #########################################################################
    # Utility functions
    #########################################################################
    def summary(self):
        print("Input shape: ", self.input_size)
        return summary(self, input_size=self.input_size, device=self.device)

    def __str__(self):
        return self.name


class Generator(NeuralNetwork):
    def __init__(self, network, input_size, device, ngpu):
        super().__init__(network, input_size=input_size, name="Generator", device=device, ngpu=ngpu)


class Adversariat(NeuralNetwork):
    """ Implements adversariat architecture.

    Might either be a discriminator (output [0, 1]) or critic (output [-Inf, Inf]).
    """
    def __init__(self, network, input_size, adv_type, device, ngpu):
        try:
            last_layer_type = type(network[-1])
        except TypeError:
            last_layer_type = type(network.__dict__["_modules"]["output"])

        valid_types = ["Discriminator", "Critic", "AutoEncoder"]
        if adv_type == "Discriminator":
            valid_last_layer = [torch.nn.Sigmoid]
        elif adv_type == "Critic":
            valid_last_layer = [torch.nn.Linear, torch.nn.Identity]
        elif adv_type == "AutoEncoder":
            valid_last_layer = []
        else:
            raise TypeError("adv_type must be one of {}.".format(valid_types))
        self._type = adv_type

        if len(valid_last_layer) > 0:
            assert last_layer_type in valid_last_layer, (
                "Last layer activation function of {} needs to be one of '{}'.".format(adv_type, valid_last_layer)
            )

        super().__init__(network, input_size=input_size, name="Adversariat", device=device, ngpu=ngpu)

    def predict(self, x):
        return self(x)


class Encoder(NeuralNetwork):
    def __init__(self, network, input_size, device, ngpu):
        valid_last_layer = [torch.nn.Linear, torch.nn.Identity]
        try:
            last_layer_type = type(network[-1])
        except TypeError:
            last_layer_type = type(network.__dict__["_modules"]["output"])
        assert last_layer_type in valid_last_layer, (
            "Last layer activation function of Encoder needs to be one of '{}'.".format(valid_last_layer)
        )
        super().__init__(network, input_size=input_size, name="Encoder", device=device, ngpu=ngpu)


class Decoder(NeuralNetwork):
    def __init__(self, network, input_size, device, ngpu):
        super().__init__(network, input_size=input_size, name="Decoder", device=device, ngpu=ngpu)


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()