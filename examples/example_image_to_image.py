import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading

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
    X_train, y_train, X_test, y_test = loading.load_data(datapath, which="mnist", download=True)
    y_train = np.array([np.rot90(im) for im in X_train])
    y_test = np.array([np.rot90(im) for im in X_test])

    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 5
    batch_size = 16
    optim_kwargs = {"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    y_train = y_train.reshape((-1, 1, 32, 32))
    y_test = y_test.reshape((-1, 1, 32, 32))
    x_dim = X_train.shape[1:]
    y_dim = y_train.shape[1:]
    z_dim = 32
    gen_in_dim = utils.get_input_dim(dim1=z_dim, dim2=y_dim)
    adv_in_dim = utils.get_input_dim(dim1=x_dim, dim2=y_dim)

    ######################################C###################################
    # Architecture
    #########################################################################
    generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="example")
    discriminator = loading.load_adversariat(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator", which="example")
    critic = loading.load_adversariat(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic", which="example")
    encoder = loading.load_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="example")
    autoencoder = loading.load_autoencoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="example")
    decoder = loading.load_decoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="example")

    #########################################################################
    # Training
    #########################################################################

    # gan_model = ConditionalKLGAN(
    #     generator=generator, adversariat=discriminator,
    #     x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/CGAN", optim=None,
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16
    # )

    gan_model = ConditionalLRGAN(
        generator=generator, adversariat=discriminator, encoder=encoder,
        x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/CGAN", optim=None,
        optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16
    )

    # gan_model = ConditionalEBGAN(
    #     generator=generator, adversariat=autoencoder,
    #     x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/EBGAN", optim=None,
    #     optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}, fixed_noise_size=16, m=np.mean(X_train)
    # )

    # gan_model = ConditionalVanillaVAE(
    #     encoder=encoder, decoder=decoder,
    #     z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, folder="TrainedModels/VAEGAN",}
    # )

    # gan_model = ConditionalBicycleGAN(
    #     encoder=encoder, generator=generator, adversariat=critic,
    #     z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, folder="TrainedModels/VAEGAN", adv_type="Critic",
    #     optim_kwargs={"Generator": {"lr": 0.001}, "Adversariat": {"lr": 0.0005}}
    # )

    # gan_model = ConditionalAAE(
    #     encoder=encoder, generator=generator,
    #     adversariat=loading.load_adversariat(x_dim=z_dim, z_dim=None, y_dim=y_dim, adv_type="Critic", which="example"),
    #     z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, folder="TrainedModels/cAAE", adv_type="Critic",
    #     optim_kwargs={"Generator": {"lr": 0.001}, "Adversariat": {"lr": 0.0005}}
    # )

    # gan_model = ConditionalPix2Pix(
    #     generator=generator, adversariat=adversariat,
    #     x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder="TrainedModels/Im2Im",
    #     optim_kwargs=optim_kwargs, fixed_noise_size=16
    # )

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
        save_model_every=None,
        save_images_every="0.5e",
        save_losses_every=10,
        enable_tensorboard=True,
    )
    samples, losses = gan_model.get_training_results(by_epoch=False)
    gan_model.save()