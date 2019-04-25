# MAC fix
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from vegans import BEGAN, WGANGP

# general params
image_size = 32
num_channels = 1
img_shape = (num_channels, image_size, image_size)
latent_dim = 64


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
        # shuffle=False to have some stability between different models' training
        super().__init__(dataset, batch_size=64, shuffle=False)


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


# class WGANGPGenerator(nn.Module):
#     def __init__(self, img_shape, latent_dim):
#         super(WGANGPGenerator, self).__init__()
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *block(latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.shape[0], *img_shape)
#
#         return img
#
#
# class WGANGPDiscriminator(nn.Module):
#     def __init__(self, img_shape):
#         super(WGANGPDiscriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#         )
#
#     def forward(self, img):
#         img_flat = img.view(img.shape[0], -1)
#         validity = self.model(img_flat)
#
#         return validity


def train_began(**kwargs):
    gan = BEGAN(dataloader=MNISTLoader('../../data', image_size),
                discriminator=Discriminator(image_size, num_channels),
                generator=Generator(image_size, num_channels, latent_dim),
                nz=latent_dim,
                **kwargs)
    gan.train(lr_decay_every=3000)

    return gan


def train_wgan_gp(**kwargs):
    gan = WGANGP(dataloader=MNISTLoader('../../data', image_size),
                 discriminator=Discriminator(image_size, num_channels),
                 generator=Generator(image_size, num_channels, latent_dim),
                 nz=latent_dim,
                 **kwargs)
    gan.train()

    return gan


def plot_single_gif(samples, filename):
    fig, axes = plt.subplots()

    # remove axes
    plt.axis("off")
    fig.subplots_adjust(left=0, bottom=0.02, right=1, top=0.90, wspace=None, hspace=None)

    stages = list(samples.keys())
    samples = list(samples.values())

    def prepare_image(sample):
        return np.transpose(make_grid(sample, padding=2, normalize=True), (1, 2, 0))

    # init things to show
    image = (axes.imshow(prepare_image(samples[0])), axes.text(10, -10, '', fontsize=16))

    def init():
        return image

    def run(it):
        image[0].set_data(prepare_image(samples[it]))
        image[1].set_text('{} Epoch {} Batch {}'.format(filename.upper(), stages[it][0], stages[it][1]))

        return image

    anim = FuncAnimation(fig, run, frames=len(samples), init_func=init, interval=30, blit=True)
    anim.save('../resources/{}.gif'.format(filename), dpi=80, writer='imagemagick', fps=1)


def save_gif(trained_gan, name):
    samples, _, _ = trained_gan.get_training_results()
    plot_single_gif(samples, name)


def train_gans(**kwargs):
    wgan_gp = train_wgan_gp(**kwargs)
    began = train_began(**kwargs)

    save_gif(wgan_gp, 'wgan_gp')
    save_gif(began, 'began')


train_gans(device='cpu', ngpu=0)
