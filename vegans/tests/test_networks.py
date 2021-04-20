import torch
import pytest

import vegans.utils.networks as network

def generate_net(in_dim, last_layer):
    class MyNetwork(torch.nn.Module):
        def __init__(self, in_dim, last_layer):
            super().__init__()
            self.hidden_part = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 64),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(64, 10),
            )
            self.output = last_layer()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    return MyNetwork(in_dim, last_layer)

def test_NeuralNetwork():
    net = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid)
    network.NeuralNetwork(net, "Discriminator", 10, "cpu", 0, True)

    net = generate_net(in_dim=10, last_layer=torch.nn.ReLU)
    network.NeuralNetwork(net, "Discriminator", 10, "cpu", 0, True)

    net = generate_net(in_dim=10, last_layer=torch.nn.Tanh)
    net = network.NeuralNetwork(net, "Something", 10, "cpu", 3, True)
    assert net.output_size == (10, )

    with pytest.raises(TypeError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Tanh)
        network.NeuralNetwork(net, "Something", 11, "cpu", 3, True)


def test_Adversariat():
    net = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid)
    network.Adversariat(net, 10, "Discriminator", "cpu", 0, True)

    net = generate_net(in_dim=10, last_layer=torch.nn.Identity)
    network.Adversariat(net, 10, "Critic", "cpu", 0, True)

    with pytest.raises(AssertionError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid)
        network.Adversariat(net, 10, "Critic", "cpu", 0, True)

    with pytest.raises(AssertionError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Identity)
        network.Adversariat(net, 10, "Discriminator", "cpu", 0, True)

def test_Encoder():
    net = generate_net(in_dim=10, last_layer=torch.nn.Identity)
    network.Encoder(net, 10, "cpu", 0, True)

    with pytest.raises(AssertionError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Tanh)
        network.Encoder(net, 10, "cpu", 0, True)