import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

from vegans.GAN import (
    AAE,
    BicycleGAN,
    EBGAN,
    LSGAN,
    LRGAN,
    VanillaGAN,
    WassersteinGAN,
    WassersteinGANGP,
    VAEGAN,
)
from vegans.models.unconditional.VanillaVAE import VanillaVAE

if __name__ == '__main__':

    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = loading.load_data(datapath, which="mnist", download=True)
    lr_gen = 0.0001
    lr_adv = 0.0001
    epochs = 2
    batch_size = 32

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    x_dim = X_train.shape[1:]
    z_dim = 128

    ######################################C###################################
    # Architecture
    #########################################################################
    generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, which="example")
    discriminator = loading.load_adversariat(x_dim=x_dim, z_dim=z_dim, adv_type="Discriminator", which="example")
    critic = loading.load_adversariat(x_dim=x_dim, z_dim=z_dim, adv_type="Critic", which="example")
    encoder = loading.load_encoder(x_dim=x_dim, z_dim=z_dim, which="example")
    autoencoder = loading.load_autoencoder(x_dim=x_dim, z_dim=z_dim, which="example")
    decoder = loading.load_decoder(x_dim=x_dim, z_dim=z_dim, which="example")

    #########################################################################
    # Training
    #########################################################################

    gan_model = WassersteinGAN(
        generator=generator, adversariat=critic,
        z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/GAN",
        feature_layer=discriminator.hidden_part,
        optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}
    )

    # gan_model = LRGAN(
    #     generator=generator, adversariat=discriminator, encoder=encoder,
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/GAN",
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}
    # )

    # gan_model = EBGAN(
    #     generator=generator, decoder=autoencoder,
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/GAN",
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, m=np.mean(X_train)
    # )

    # gan_model = VanillaVAE(
    #     encoder=encoder, decoder=decoder,
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/VAEGAN"
    # )

    # gan_model = BicycleGAN(
    #     encoder=encoder, generator=generator, adversariat=discriminator,
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/VAEGAN",
    #     optim_kwargs={"Generator": {"lr": 0.001}, "Adversariat": {"lr": 0.0005}}
    # )

    # gan_model = AAE(
    #     encoder=encoder, generator=generator,
    #     adversariat=loading.load_adversariat(x_dim=z_dim, z_dim=None, adv_type="Discriminator", which="example"),
    #     z_dim=z_dim, x_dim=x_dim, folder="TrainedModels/AAE",
    #     optim_kwargs={"Generator": {"lr": 0.001}, "Adversariat": {"lr": 0.0005}}
    # )

    gan_model.summary(save=True)
    gan_model.fit(
        X_train=X_train,
        X_test=X_test,
        batch_size=batch_size,
        epochs=epochs,
        steps={"Generator": 1, "Adversariat": 3},
        print_every=100,
        save_model_every=None,
        save_images_every="0.25e",
        save_losses_every=1,
        enable_tensorboard=True
    )
    samples, losses = gan_model.get_training_results(by_epoch=False)
    utils.plot_images(images=samples.reshape(-1, 32, 32))
    utils.plot_losses(losses=losses)
    # gan_model.save()

