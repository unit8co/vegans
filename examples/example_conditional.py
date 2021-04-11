import torch

import numpy as np
import vegans.utils.utils as utils

from torch import nn
from sklearn.preprocessing import OneHotEncoder
from vegans.utils.layers import LayerReshape, LayerPrintSize
from vegans.GAN import ConditionalVanillaGAN, ConditionalWassersteinGAN, ConditionalWassersteinGANGP, ConditionalLRGAN


if __name__ == '__main__':

    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)

    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 2
    batch_size = 32

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

    im_dim = X_train.shape[1:]
    y_dim = y_train.shape[1:]
    #########################################################################
    # Flat network
    #########################################################################
    z_dim = 128
    gen_in_dim = utils.get_input_dim(dim1=z_dim, dim2=y_dim)
    adv_in_dim = utils.get_input_dim(dim1=im_dim, dim2=y_dim)

    class MyGenerator(nn.Module):
        def __init__(self, z_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(gen_in_dim), 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, int(np.prod(im_dim))),
                LayerReshape(im_dim)
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    class MyAdversariat(nn.Module):
        def __init__(self, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Conv2d(in_channels=x_dim[0], out_channels=1, kernel_size=4, stride=4, padding=0),
                nn.Flatten(),
                nn.Linear(64, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1)
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    class MyEncoder(nn.Module):
        def __init__(self, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(np.prod(x_dim)), 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, np.prod(z_dim))
            )
            self.output = nn.Identity()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    #########################################################################
    # Training
    #########################################################################
    generator = MyGenerator(z_dim=z_dim)
    adversariat = MyAdversariat(x_dim=im_dim)
    encoder = MyEncoder(x_dim=im_dim)

    # gan_model = ConditionalWassersteinGAN(
    #     generator=generator, adversariat=adversariat,
    #     x_dim=im_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/CGAN", optim=None,
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16
    # )
    gan_model = ConditionalLRGAN(
        generator=generator, adversariat=adversariat, encoder=encoder,
        x_dim=im_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/CGAN", optim=None,
        optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16
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
        enable_tensorboard=True
    )
    samples, losses = gan_model.get_training_results(by_epoch=False)
    utils.plot_images(images=samples.reshape(-1, 32, 32))
    utils.plot_losses(losses=losses)
    # gan_model.save()