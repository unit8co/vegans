import numpy as np
import vegans.utils.utils as utils
import vegans.utils.loading as loading
import matplotlib.pyplot as plt

from vegans.GAN import (
    AAE,
    BicycleGAN,
    EBGAN,
    InfoGAN,
    KLGAN,
    LRGAN,
    LSGAN,
    VAEGAN,
    VanillaGAN,
    VanillaVAE,
    WassersteinGAN,
    WassersteinGANGP,
)
from vegans.models.unconditional.VanillaVAE import VanillaVAE

if __name__ == '__main__':

    loader = loading.MNISTLoader()
    X_train, _, X_test, _ = loader.load()
    X_train = X_train[:500]
    epochs = 1
    batch_size = 32

    x_dim = X_train.shape[1:]
    z_dim = 2
    y_dim = None

    ######################################C###################################
    # Architecture
    #########################################################################
    # loader = loading.ExampleLoader()
    generator = loader.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    discriminator = loader.load_adversary(x_dim=x_dim, y_dim=y_dim, adv_type="Discriminator")
    critic = loader.load_adversary(x_dim=x_dim, y_dim=y_dim, adv_type="Critic")
    encoder = loader.load_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    autoencoder = loader.load_autoencoder(x_dim=x_dim, y_dim=y_dim)
    decoder = loader.load_decoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    #########################################################################
    # Training
    #########################################################################
    models = [
        # AAE, BicycleGAN, EBGAN,
        # InfoGAN, KLGAN, LRGAN, LSGAN,
        # VAEGAN, VanillaGAN,
        # VanillaVAE , WassersteinGAN, WassersteinGANGP,
        LRGAN
    ]

    for model in models:
        kwargs = {"x_dim": x_dim, "z_dim": z_dim}

        if model.__name__ in ["AAE"]:
            discriminator_aee = loading.ExampleLoader().load_adversary(x_dim=z_dim, y_dim=y_dim, adv_type="Discriminator")
            gan_model = model(
                generator=generator, adversary=discriminator_aee, encoder=encoder, **kwargs
            )

        elif model.__name__ in ["BicycleGAN", "VAEGAN"]:
            encoder_reduced = loader.load_encoder(x_dim=x_dim, z_dim=z_dim*2, y_dim=y_dim)
            gan_model = model(
                generator=generator, adversary=discriminator, encoder=encoder_reduced, **kwargs
            )

        elif model.__name__ in ["EBGAN"]:
            m = np.mean(X_train)
            gan_model = model(
                generator=generator, adversary=autoencoder, m=m, **kwargs
            )

        elif model.__name__ in ["InfoGAN"]:
            c_dim_discrete = [10]
            c_dim_continuous = 0
            c_dim = sum(c_dim_discrete) + c_dim_continuous
            generator_conditional = loader.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=c_dim)
            encoder_helper = loader.load_encoder(x_dim=x_dim, z_dim=32)
            gan_model = model(
                generator=generator_conditional, adversary=discriminator, encoder=encoder_helper,
                c_dim_discrete=c_dim_discrete, c_dim_continuous=c_dim_continuous, **kwargs
            )

        elif model.__name__ in ["KLGAN", "LSGAN", "VanillaGAN"]:
            gan_model = model(
                generator=generator, adversary=discriminator, **kwargs
            )

        elif model.__name__ in ["LRGAN"]:
            gan_model = model(
                generator=generator, adversary=discriminator, encoder=encoder, **kwargs
            )

        elif model.__name__ in ["VanillaVAE"]:
            encoder_reduced = loader.load_encoder(x_dim=x_dim, z_dim=z_dim*2, y_dim=y_dim)
            gan_model = model(
                encoder=encoder_reduced, decoder=decoder, **kwargs
            )

        elif model.__name__ in ["WassersteinGAN", "WassersteinGANGP"]:
            gan_model = model(
                generator=generator, adversary=critic, **kwargs
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
            save_images_every="0.5e",
            save_losses_every=10,
            enable_tensorboard=True
        )
        samples, losses = gan_model.get_training_results(by_epoch=False)

        training_time = np.round(gan_model.total_training_time/60, 2)
        title = "Epochs: {}, z_dim: {}, Time trained: {} minutes\nParams: {}\n\n".format(
            epochs, z_dim, training_time, gan_model.get_number_params()
        )
        fig, axs = utils.plot_images(images=samples.reshape(-1, 32, 32), show=False)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        plt.savefig(gan_model.folder+"/generated_images.png")

        fig, axs = utils.plot_losses(losses=losses, show=False)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        plt.savefig(gan_model.folder+"/losses.png")
        # gan_model.save()