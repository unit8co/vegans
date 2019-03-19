from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


_split_mapping = {
    'train': True,
    'test': False
}


def mnist_data_loader(path, split='train', batch_size=32):
    return DataLoader(
        MNIST(path,
              train=_split_mapping[split],
              download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              ])),
        batch_size=batch_size, shuffle=True)
