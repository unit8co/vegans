import numpy as np
import pytest
import torch.nn.functional as f
from torch import nn, sigmoid
from torch.utils.data import DataLoader, Dataset


class GaussianDataset(Dataset):
    def __init__(self, sigma=1.0, mu=0.0):
        self.sigma = sigma
        self.mu = mu

    def __getitem__(self, idx):
        return np.array([self.sigma * np.random.randn() + self.mu]).astype(np.float32), 0

    def __len__(self):
        return 1000


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(nz, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = sigmoid(self.fc2(x))

        return x


nz = 5


@pytest.fixture(scope='module')
def gaussian_dataloader():
    return DataLoader(GaussianDataset(), batch_size=nz)


@pytest.fixture(scope='module')
def generator():
    return Generator(nz=5)


@pytest.fixture(scope='module')
def critic():
    return Critic()
