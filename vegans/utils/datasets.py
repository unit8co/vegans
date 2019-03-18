import numpy as np
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    def __init__(self, mu, sigma, seed=None):
        self.mu = mu
        self.sigma = sigma
        if seed:
            np.random.seed(seed)

    def __getitem__(self, idx):
        return np.array([self.sigma * np.random.randn() + self.mu]).astype(np.float32), 0

    def __len__(self):
        return 10000
