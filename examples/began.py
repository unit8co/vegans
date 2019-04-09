from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from vegans import BEGAN


class MNISTLoader(DataLoader):
    def __init__(self, path, img_size, train=True):
        dataset = MNIST(path,
                        train=train,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))
        super().__init__(dataset, batch_size=64, shuffle=64)


class Generator(nn.Module):
    def __init__(self, img_size, channels, latent_dim):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 2, 1),
            nn.ReLU(),
        )
        # Fully-connected layers
        self.down_size = (img_size // 2)
        down_dim = 64 * (img_size // 2)**2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True)
        )
        # Upsampling
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, channels, 3, 1, 1)
        )

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))

        return out


img_size = 32
channels = 1
nz = 62

gan = BEGAN(dataloader=MNISTLoader('../../data', img_size),
            discriminator=Discriminator(img_size, channels),
            generator=Generator(img_size, channels, nz),
            nz=nz)
gan.train(lr_decay_every=3000)
