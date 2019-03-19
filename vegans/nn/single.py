import torch
import torch.nn as nn
import torch.nn.functional as f


class SingleLayerNNReLU(nn.Module):
    def __init__(self, input_size=32, hidden_size=10, num_classes=1):
        super(SingleLayerNNReLU, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)

        return x


class SingleLayerNNSigmoid(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_classes=1):
        super(SingleLayerNNSigmoid, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)

        return x
