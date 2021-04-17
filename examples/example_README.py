mode = "unsupervised"

if mode == "unsupervised":
    from vegans.GAN import WassersteinGAN
    import vegans.utils.utils as utils
    import vegans.utils.loading as loading

    datapath =  "./data/mnist/"
    X_train, y_train, X_test, y_test = (
        loading.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    )
    X_train = X_train.reshape((-1, 1, 32, 32)) # required shape
    X_test = X_test.reshape((-1, 1, 32, 32))
    x_dim = X_train.shape[1:] # [nr_channels, height, width]
    z_dim = 64

    # Define your own architectures here. You can use a Sequential model or an object
    # inheriting from torch.nn.Module.
    generator = loading.load_example_generator(x_dim=x_dim, z_dim=z_dim)
    critic = loading.load_example_adversariat(x_dim=x_dim, z_dim=z_dim, adv_type="Critic")

    gan = WassersteinGAN(
        generator=generator, adversariat=critic,
        z_dim=z_dim, x_dim=x_dim, folder=None
    )
    gan.summary() # optional, shows architecture
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
    from vegans.GAN import ConditionalWassersteinGAN
    from sklearn.preprocessing import OneHotEncoder # Download sklearn

    datapath =  "./data/mnist/"
    X_train, y_train, X_test, y_test = (
        loading.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    )
    X_train = X_train.reshape((-1, 1, 32, 32)) # required shape
    X_test = X_test.reshape((-1, 1, 32, 32))
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

    x_dim = X_train.shape[1:] # [nr_channels, height, width]
    y_dim = y_train.shape[1:]
    z_dim = 64

    # Define your own architectures here. You can use a Sequential model or an object
    # inheriting from torch.nn.Module.
    generator = loading.load_example_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
    critic = loading.load_example_adversariat(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic")

    gan = ConditionalWassersteinGAN(
        generator=generator, adversariat=critic,
        z_dim=z_dim, x_dim=x_dim, y_dim=y_dim,
        folder=None, # optional
        optim={"Generator": torch.optim.RMSprop, "Adversariat": torch.optim.Adam}, # optional
        optim_kwargs={"Generator": {"lr": 0.0001}, "Adversariat": {"lr": 0.0001}}, # optional
        fixed_noise_size=32, # optional
        clip_val=0.01, # optional
        device=None, # optional
        ngpu=0 # optional

    )
    gan.summary() # optional, shows architecture
    gan.fit(
        X_train, y_train, X_test, y_test,
        epochs=5, # optional
        batch_size=32, # optional
        steps={"Generator": 1, "Adversariat": 5}, # optional
        print_every="0.1e", # optional
        save_model_every=None, # optional
        save_images_every=None, # optional
        save_losses_every="0.1e", # optional
        enable_tensorboard=False # optional
    )

    # Vizualise results
    images, losses = gan.get_training_results()
    images = images.reshape(-1, *images.shape[2:]) # remove nr_channels for plotting
    utils.plot_images(images, labels=np.argmax(gan.fixed_labels.cpu().numpy(), axis=1))
    utils.plot_losses(losses)