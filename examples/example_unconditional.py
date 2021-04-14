import torch

import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

from torch import nn
from vegans.utils.layers import LayerReshape, LayerPrintSize
from vegans.GAN import (
    VanillaGAN, WassersteinGAN, WassersteinGANGP,
    LSGAN, LRGAN, EBGAN
)
from vegans.models.unconditional.VanillaVAE import VanillaVAE

if __name__ == '__main__':

    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = loading.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 2
    batch_size = 64

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    x_dim = X_train.shape[1:]
    z_dim = 128


    ######################################C###################################
    # Architecture
    #########################################################################
    generator = loading.load_example_generator(x_dim=x_dim, z_dim=z_dim)
    adversariat = loading.load_example_adversariat(x_dim=x_dim, z_dim=z_dim, adv_type="Critic")
    encoder = loading.load_example_encoder(x_dim=x_dim, z_dim=z_dim)
    autoencoder = loading.load_example_autoencoder(x_dim=x_dim, z_dim=z_dim)
    decoder = loading.load_example_decoder(x_dim=x_dim, z_dim=z_dim)

    #########################################################################
    # Training
    #########################################################################

    # gan_model = WassersteinGAN(
    #     generator=generator, adversariat=adversariat,
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/GAN", optim={"Generator": torch.optim.Adam},
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}
    # )

    # gan_model = LRGAN(
    #     generator=generator, adversariat=adversariat, encoder=encoder,
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/GAN", optim={"Generator": torch.optim.Adam},
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}
    # )

    # gan_model = EBGAN(
    #     generator=generator, decoder=autoencoder,
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/GAN", optim={"Generator": torch.optim.Adam},
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, m=np.mean(X_train)
    # )

    gan_model = VanillaVAE(
        encoder=encoder, decoder=decoder,
        z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/VAE", optim={"Autoencoder": torch.optim.Adam}
    )
    gan_model.summary(save=True)
    gan_model.fit(
        X_train=X_train,
        X_test=X_test,
        batch_size=batch_size,
        epochs=epochs,
        steps=None,
        print_every=100,
        save_model_every="3e",
        save_images_every="0.25e",
        save_losses_every=1,
        enable_tensorboard=True
    )
    samples, losses = gan_model.get_training_results(by_epoch=False)
    utils.plot_images(images=samples.reshape(-1, 32, 32))
    utils.plot_losses(losses=losses)
    # gan_model.save()

