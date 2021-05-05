mode = "supervised"

if mode == "unsupervised":
    from vegans.GAN import VanillaGAN
    import vegans.utils.utils as utils
    import vegans.utils.loading as loading

    # Data preparation
    datapath =  "./data/" # https://github.com/tneuer/vegans/tree/version/overhaul/data/mnist
    X_train, y_train, X_test, y_test = loading.load_data(datapath, which="mnist", download=True)
    X_train = X_train.reshape((-1, 1, 32, 32)) # required shape
    X_test = X_test.reshape((-1, 1, 32, 32))
    x_dim = X_train.shape[1:] # [height, width, nr_channels]
    z_dim = 64

    # Define your own architectures here. You can use a Sequential model or an object
    # inheriting from torch.nn.Module. Here, a default model for mnist is loaded.
    generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, which="example")
    discriminator = loading.load_adversary(x_dim=x_dim, z_dim=z_dim, adv_type="Discriminator", which="example")

    gan = VanillaGAN(
        generator=generator, adversary=discriminator,
        z_dim=z_dim, x_dim=x_dim, folder=None
    )
    gan.summary() # optional, shows architecture

    # Training
    gan.fit(X_train, enable_tensorboard=False)

    # Vizualise results
    images, losses = gan.get_training_results()
    images = images.reshape(-1, *images.shape[2:]) # remove nr_channels for plotting
    utils.plot_images(images)
    utils.plot_losses(losses)

    # Sample new images, you can also pass a specific noise vector
    samples = gan.generate(n=36)
    samples = samples.reshape(-1, *samples.shape[2:]) # remove nr_channels for plotting
    utils.plot_images(samples)

elif mode == "supervised":
    import torch
    import numpy as np
    import vegans.utils.utils as utils
    import vegans.utils.loading as loading
    from vegans.GAN import ConditionalVanillaGAN

    # Data preparation
    datapath =  "./data/" # https://github.com/tneuer/vegans/tree/version/overhaul/data/mnist
    X_train, y_train, X_test, y_test = loading.load_data(datapath, which="mnist", download=True)
    X_train = X_train.reshape((-1, 1, 32, 32)) # required shape
    X_test = X_test.reshape((-1, 1, 32, 32))
    nb_classes = len(set(y_train))
    y_train = np.eye(nb_classes)[y_train.reshape(-1)]
    y_test = np.eye(nb_classes)[y_test.reshape(-1)]

    x_dim = X_train.shape[1:] # [nr_channels, height, width]
    y_dim = y_train.shape[1:]
    z_dim = 64

    # Define your own architectures here. You can use a Sequential model or an object
    # inheriting from torch.nn.Module. Here, a default model for mnist is loaded.
    generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")
    discriminator = loading.load_adversary(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator", which="mnist")

    gan = ConditionalVanillaGAN(
        generator=generator, adversary=discriminator,
        z_dim=z_dim, x_dim=x_dim, y_dim=y_dim,
        folder=None, # optional
        optim={"Generator": torch.optim.RMSprop, "Adversary": torch.optim.Adam}, # optional
        optim_kwargs={"Generator": {"lr": 0.0001}, "Adversary": {"lr": 0.0001}}, # optional
        fixed_noise_size=32, # optional
        device=None, # optional
        ngpu=0 # optional

    )
    gan.summary() # optional, shows architecture

    # Training
    gan.fit(
        X_train, y_train, X_test, y_test,
        epochs=5, # optional
        batch_size=32, # optional
        steps={"Generator": 1, "Adversary": 2}, # optional, train generator once and discriminator twice on every mini-batch
        print_every="0.1e", # optional, prints progress 10 times per epoch
                            # (might also be integer input indicating number of mini-batches)
        save_model_every=None, # optional
        save_images_every=None, # optional
        save_losses_every="0.1e", # optional, save losses in internal losses dictionary used to generate
                                  # plots during and after training
        enable_tensorboard=False # optional, if true all progress is additionally saved in tensorboard subdirectory
    )

    # Vizualise results
    images, losses = gan.get_training_results()
    images = images.reshape(-1, *images.shape[2:]) # remove nr_channels for plotting
    utils.plot_images(images, labels=np.argmax(gan.fixed_labels.cpu().numpy(), axis=1))
    utils.plot_losses(losses)

    # Generate specific label, for example "2"
    label = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0,0 ]])
    image = gan(y=label)
    utils.plot_images(image, labels=["2"])
