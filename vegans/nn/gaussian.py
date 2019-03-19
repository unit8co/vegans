import torch
import torch.nn as nn
import torch.nn.functional as f


class GaussianGenerator(nn.Module):
    def __init__(self, input_size=32, hidden_layer_size=10):
        super(GaussianGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class GaussianDiscriminator(nn.Module):
    def __init__(self, hidden_layer_size=10, output_size=1):
        super(GaussianDiscriminator, self).__init__()
        self.fc1 = nn.Linear(1, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
