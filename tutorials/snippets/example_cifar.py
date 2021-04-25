import torch

import numpy as np
import matplotlib.pyplot as plt
import vegans.utils.utils as utils
import vegans.utils.loading as loading

from vegans.GAN import (
    ConditionalAAE,
    ConditionalBicycleGAN,
    ConditionalEBGAN,
    ConditionalKLGAN,
    ConditionalLRGAN,
    ConditionalLSGAN,
    ConditionalPix2Pix,
    ConditionalVAEGAN,
    ConditionalVanillaGAN,
    ConditionalVanillaVAE,
    ConditionalWassersteinGAN,
    ConditionalWassersteinGANGP,
)
from vegans.models.conditional.ConditionalVanillaVAE import ConditionalVanillaVAE


if __name__ == '__main__':

    datapath = "./data/"
    X_train, y_train, X_test, y_test = loading.load_data(datapath, which="cifar", download=True)

    epochs = 3
    batch_size = 32

    X_train = X_train.reshape((-1, 3, 32, 32))
    X_test = X_test.reshape((-1, 3, 32, 32))
    nb_classes = len(set(y_train))
    y_train = np.eye(nb_classes)[y_train.reshape(-1)]
    y_test = np.eye(nb_classes)[y_test.reshape(-1)]

    x_dim = X_train.shape[1:]
    y_dim = y_train.shape[1:]
    z_dim = 64
    gen_in_dim = utils.get_input_dim(dim1=z_dim, dim2=y_dim)
    adv_in_dim = utils.get_input_dim(dim1=x_dim, dim2=y_dim)

    ######################################C###################################
    # Architecture
    #########################################################################
    generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")
    discriminator = loading.load_adversary(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator", which="mnist")
    critic = loading.load_adversary(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic", which="mnist")
    encoder = loading.load_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")
    autoencoder = loading.load_autoencoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")
    decoder = loading.load_decoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")

    #########################################################################
    # Training
    #########################################################################
    models = [
        # ConditionalAAE, ConditionalBicycleGAN, ConditionalEBGAN,
        # ConditionalKLGAN, ConditionalLRGAN, ConditionalLSGAN,
        # ConditionalPix2Pix, ConditionalVAEGAN, ConditionalVanillaGAN,
        # ConditionalVanillaVAE , ConditionalWassersteinGAN, ConditionalWassersteinGANGP,
        ConditionalWassersteinGAN, ConditionalWassersteinGANGP,
    ]

    for model in models:
        folder = "MyModels/{}".format(model.__name__.replace("Conditional", "c"))
        kwargs = {"x_dim": x_dim, "z_dim": z_dim, "y_dim": y_dim, "folder": folder}

        if model.__name__ in ["ConditionalAAE"]:
            discriminator_aee = loading.load_adversary(x_dim=z_dim, z_dim=None, y_dim=y_dim, adv_type="Discriminator", which="example")
            gan_model = model(
                generator=generator, adversary=discriminator_aee, encoder=encoder, **kwargs
            )

        elif model.__name__ in ["ConditionalBicycleGAN", "ConditionalVAEGAN"]:
            encoder_reduced = loading.load_encoder(x_dim=x_dim, z_dim=z_dim//2, y_dim=y_dim, which="mnist")
            gan_model = model(
                generator=generator, adversary=discriminator, encoder=encoder_reduced, **kwargs
            )

        elif model.__name__ in ["ConditionalEBGAN"]:
            m = np.mean(X_train)
            gan_model = model(
                generator=generator, adversary=autoencoder, m=m, **kwargs
            )

        elif model.__name__ in ["ConditionalKLGAN", "ConditionalLSGAN", "ConditionalPix2Pix", "ConditionalVanillaGAN"]:
            gan_model = model(
                generator=generator, adversary=discriminator, **kwargs
            )

        elif model.__name__ in ["ConditionalLRGAN"]:
            gan_model = model(
                generator=generator, adversary=discriminator, encoder=encoder, **kwargs
            )

        elif model.__name__ in ["ConditionalVanillaVAE"]:
            encoder_reduced = loading.load_encoder(x_dim=x_dim, z_dim=z_dim//2, y_dim=y_dim, which="mnist")
            gan_model = model(
                encoder=encoder_reduced, decoder=decoder, **kwargs
            )

        elif model.__name__ in ["ConditionalWassersteinGAN", "ConditionalWassersteinGANGP"]:
            gan_model = model(
                generator=generator, adversary=critic, **kwargs
            )

        else:
            raise NotImplementedError("{} no yet implemented in logical gate.".format(model.__name__))

        gan_model.summary(save=True)
        gan_model.fit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
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