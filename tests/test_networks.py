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
    network.NeuralNetwork(network=net, name="Discriminator", input_size=10, device="cpu", ngpu=0, secure=True)

    net = generate_net(in_dim=10, last_layer=torch.nn.ReLU)
    network.NeuralNetwork(network=net, name="Discriminator", input_size=10, device="cpu", ngpu=0, secure=True)

    net = generate_net(in_dim=10, last_layer=torch.nn.Tanh)
    net = network.NeuralNetwork(network=net, name="Something", input_size=10, device="cpu", ngpu=3, secure=True)
    assert net.output_size == (10, )

    with pytest.raises(TypeError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Tanh)
        network.NeuralNetwork(network=net, name="Something", input_size=11, device="cpu", ngpu=3, secure=True)


def test_Adversary():
    net = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid)
    network.Adversary(network=net, input_size=10, adv_type="Discriminator", device="cpu", ngpu=0, secure=True)

    net = generate_net(in_dim=10, last_layer=torch.nn.Identity)
    network.Adversary(network=net, input_size=10, adv_type="Critic", device="cpu", ngpu=0, secure=True)

    with pytest.raises(AssertionError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Sigmoid)
        network.Adversary(network=net, input_size=10, adv_type="Critic", device="cpu", ngpu=0, secure=True)

    with pytest.raises(AssertionError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Identity)
        network.Adversary(network=net, input_size=10, adv_type="Discriminator", device="cpu", ngpu=0, secure=True)

def test_Encoder():
    net = generate_net(in_dim=10, last_layer=torch.nn.Identity)
    network.Encoder(network=net, input_size=10, device="cpu", ngpu=0, secure=True)

    with pytest.raises(AssertionError):
        net = generate_net(in_dim=10, last_layer=torch.nn.Tanh)
        network.Encoder(network=net, input_size=10, device="cpu", ngpu=0, secure=True)