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

    datapath = "./data/"
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
    models = [
        AAE, #BicycleGAN, EBGAN,
        # KLGAN, LRGAN, LSGAN,
        # Pix2Pix, VAEGAN, VanillaGAN,
        # VanillaVAE , WassersteinGAN, WassersteinGANGP,
    ]

    for model in models:
        folder = "MyModels/{}".format(model.__name__.replace("", "c"))
        kwargs = {"x_dim": x_dim, "z_dim": z_dim, "folder": folder}

        if model.__name__ in ["AAE"]:
            discriminator_aee = loading.load_adversariat(x_dim=z_dim, z_dim=None, adv_type="Discriminator", which="example")
            gan_model = model(
                generator=generator, adversariat=discriminator_aee, encoder=encoder, **kwargs
            )

        elif model.__name__ in ["BicycleGAN", "VAEGAN"]:
            encoder_reduced = loading.load_encoder(x_dim=x_dim, z_dim=z_dim//2, which="mnist")
            gan_model = model(
                generator=generator, adversariat=discriminator, encoder=encoder_reduced, **kwargs
            )

        elif model.__name__ in ["EBGAN"]:
            m = np.mean(X_train)
            gan_model = model(
                generator=generator, adversariat=autoencoder, m=m, **kwargs
            )

        elif model.__name__ in ["KLGAN", "LSGAN", "Pix2Pix", "VanillaGAN"]:
            gan_model = model(
                generator=generator, adversariat=discriminator, **kwargs
            )

        elif model.__name__ in ["LRGAN"]:
            gan_model = model(
                generator=generator, adversariat=discriminator, encoder=encoder, **kwargs
            )

        elif model.__name__ in ["VanillaVAE"]:
            encoder_reduced = loading.load_encoder(x_dim=x_dim, z_dim=z_dim//2, which="mnist")
            gan_model = model(
                encoder=encoder_reduced, decoder=decoder, **kwargs
            )

        elif model.__name__ in ["WassersteinGAN", "WassersteinGANGP"]:
            gan_model = model(
                generator=generator, adversariat=critic, **kwargs
            )

        else:
            raise NotImplementedError("{} no yet implemented in logical gate.".format(model.__name__))

        gan_model.summary(save=True)
        gan_model.fit(
            X_train=X_train,
            X_test=X_test,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            print_every="0.2e",
            save_model_every=None,
            save_images_every="0.2e",
            save_losses_every=10,
            enable_tensorboard=False
        )
        samples, losses = gan_model.get_training_results(by_epoch=False)

        training_time = np.round(gan_model.total_training_time/60, 2)
        title = "Epochs: {}, z_dim: {}, Time trained: {} minutes\nParams: {}\n\n".format(
            epochs, z_dim, training_time, gan_model.get_number_params()
        )
        fixed_labels = np.argmax(gan_model.get_fixed_labels(), axis=1)
        fig, axs = utils.plot_images(images=samples.reshape(-1, 32, 32), labels=fixed_labels, show=False)
        fig.suptitle(
            title,
            fontsize=12
        )
        fig.tight_layout()
        plt.savefig(gan_model.folder+"generated_images.png")
        fig, axs = utils.plot_losses(losses=losses, show=False)
        fig.suptitle(
            title,
            fontsize=12
        )
        fig.tight_layout()
        plt.savefig(gan_model.folder+"losses.png")
        # gan_model.save()