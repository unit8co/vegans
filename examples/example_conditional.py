import torch

import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

from torch import nn
from sklearn.preprocessing import OneHotEncoder
from vegans.utils.layers import LayerReshape, LayerPrintSize
from vegans.GAN import (
    ConditionalAAE,
    ConditionalLRGAN,
    ConditionalEBGAN,
    ConditionalKLGAN,
    ConditionalVAEGAN,
    ConditionalBicycleGAN,
    ConditionalVanillaGAN,
    ConditionalWassersteinGAN,
    ConditionalWassersteinGANGP,
)
from vegans.models.conditional.ConditionalVanillaVAE import ConditionalVanillaVAE


if __name__ == '__main__':

    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = loading.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)

    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 2
    batch_size = 32

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

    x_dim = X_train.shape[1:]
    y_dim = y_train.shape[1:]
    z_dim = 128
    gen_in_dim = utils.get_input_dim(dim1=z_dim, dim2=y_dim)
    adv_in_dim = utils.get_input_dim(dim1=x_dim, dim2=y_dim)

    ######################################C###################################
    # Architecture
    #########################################################################
    generator = loading.load_example_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    discriminator = loading.load_example_adversariat(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator")
    critic = loading.load_example_adversariat(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic")
    encoder = loading.load_example_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    autoencoder = loading.load_example_autoencoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    decoder = loading.load_example_decoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    #########################################################################
    # Training
    #########################################################################

    # gan_model = ConditionalKLGAN(
    #     generator=generator, adversariat=discriminator,
    #     x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/CGAN", optim=None,
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16
    # )

    # gan_model = ConditionalLRGAN(
    #     generator=generator, adversariat=adversariat, encoder=encoder,
    #     x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/CGAN", optim=None,
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16
    # )

    # gan_model = ConditionalEBGAN(
    #     generator=generator, adversariat=autoencoder,
    #     x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/EBGAN", optim=None,
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16, m=np.mean(X_train)
    # )

    # gan_model = ConditionalVanillaVAE(
    #     encoder=encoder, decoder=decoder,
    #     z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, folder="TrainedModels/VAEGAN", optim={"Autoencoder": torch.optim.Adam}
    # )

    # gan_model = ConditionalBicycleGAN(
    #     encoder=encoder, generator=generator, adversariat=discriminator,
    #     z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, folder="TrainedModels/VAEGAN",
    #     optim_kwargs={"Generator": {"lr": 0.001}, "Adversariat": {"lr": 0.0005}}
    # )

    gan_model = ConditionalAAE(
        encoder=encoder, generator=generator,
        adversariat=loading.load_example_adversariat(x_dim=z_dim, z_dim=None, y_dim=y_dim, adv_type="Discriminator"),
        z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, folder="TrainedModels/cAAE",
        optim_kwargs={"Generator": {"lr": 0.001}, "Adversariat": {"lr": 0.0005}}
    )
    gan_model.summary(save=True)
    gan_model.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
        epochs=epochs,
        # steps={"Adversariat": 5},
        print_every=200,
        save_model_every="3e",
        save_images_every="0.25e",
        save_losses_every=10,
        enable_tensorboard=True
    )
    samples, losses = gan_model.get_training_results(by_epoch=False)
    utils.plot_images(images=samples.reshape(-1, 32, 32), labels=np.argmax(gan_model.fixed_labels.cpu().numpy(), axis=1))
    utils.plot_losses(losses=losses)
    # gan_model.save()