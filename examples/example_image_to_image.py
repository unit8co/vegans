import torch

import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

from torch import nn
from vegans.utils.layers import LayerReshape, LayerPrintSize
from vegans.GAN import ConditionalVanillaGAN, ConditionalWassersteinGAN, ConditionalWassersteinGANGP, ConditionalPix2Pix


if __name__ == '__main__':

    datapath = "../data/mnist_rotate/"
    X_train, y_train, X_test, y_test = loading.load_mnist(datapath, normalize=True, pad=0, return_datasets=False)

    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 20
    batch_size = 32

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    y_train = y_train.reshape((-1, 1, 32, 32))
    y_test = y_test.reshape((-1, 1, 32, 32))
    im_dim = X_train.shape[1:]
    label_dim = y_train.shape[1:]

    #########################################################################
    # Flat network
    #########################################################################
    z_dim = 64
    gen_in_dim = utils.get_input_dim(dim1=z_dim, dim2=label_dim)
    adv_in_dim = utils.get_input_dim(dim1=im_dim, dim2=label_dim)

    class MyGenerator(nn.Module):
        def __init__(self, z_dim):
            super().__init__()
            self.encoding = nn.Sequential(
                nn.Conv2d(in_channels=gen_in_dim[0], out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=64),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=128),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=64)
            )
            self.decoding = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=16),
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=8),
                nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.encoding(x)
            x = self.decoding(x)
            y_pred = self.output(x)
            return y_pred

    class MyAdversariat(nn.Module):
        def __init__(self, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Conv2d(in_channels=adv_in_dim[0], out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=32),
                nn.Conv2d(in_channels=32, out_channels=40, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=40),
                nn.Conv2d(in_channels=40, out_channels=20, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=20),
                nn.Conv2d(in_channels=20, out_channels=1, kernel_size=3, stride=1, padding=1),
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    #########################################################################
    # Training
    #########################################################################

    generator = MyGenerator(z_dim=z_dim)
    adversariat = MyAdversariat(x_dim=im_dim)

    optim_kwargs = {"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}

    gan_model = ConditionalPix2Pix(
        generator=generator, adversariat=adversariat,
        x_dim=im_dim, z_dim=z_dim, y_dim=label_dim, folder="TrainedModels/Im2Im", optim=torch.optim.RMSprop,
        optim_kwargs=optim_kwargs, fixed_noise_size=16
    )
    gan_model.summary(save=True)
    gan_model.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
        epochs=epochs,
        steps={"Adversariat": 5},
        print_every=200,
        save_model_every="3e",
        save_images_every="0.5",
        save_losses_every=10,
        enable_tensorboard=True,
    )
    samples, losses = gan_model.get_training_results(by_epoch=False)
    gan_model.save()