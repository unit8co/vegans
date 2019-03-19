import numpy as np
from torch.utils.data import Dataset, DataLoader


def gaussian_data_loader(batch_size=32):
    return DataLoader(GaussianDataset(), batch_size)


class GaussianDataset(Dataset):
    def __init__(self, mu=3, sigma=1, seed=None):
        self.mu = mu
        self.sigma = sigma
        if seed:
            np.random.seed(seed)

    def __getitem__(self, idx):
        return np.array([self.sigma * np.random.randn() + self.mu]).astype(np.float32), 0

    def __len__(self):
        return 10000
