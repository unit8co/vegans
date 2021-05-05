#from clearml import Task
#task = Task.init('Private cluster', 'test3')
#task.execute_remotely('gpu_support')

import numpy as np
import vegans.utils.loading as loading
from vegans.utils.utils import plot_images
from vegans.GAN import ConditionalVanillaGAN

root = "./data/"
X_train, y_train, X_test, y_test = loading.load_data(root=root, which="mnist", download=True)

X_train = X_train.reshape((-1, 1, 32, 32))[:500]
X_test = X_test.reshape((-1, 1, 32, 32))
nb_classes = len(set(y_train))
y_train = np.eye(nb_classes)[y_train.reshape(-1)]
y_test = np.eye(nb_classes)[y_test.reshape(-1)]

x_dim = X_train.shape[1:]
y_dim = y_train.shape[1:]
z_dim = 64

generator = loading.load_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, which="mnist")
discriminator = loading.load_adversary(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator", which="mnist")

gan_model = ConditionalVanillaGAN(
    generator=generator, adversary=discriminator,
    x_dim=x_dim, z_dim=z_dim, y_dim=y_dim,
    optim=None, optim_kwargs=None,                # Optional
    feature_layer=None,                           # Optional
    fixed_noise_size=32,                          # Optional
    device=None,                                  # Optional
    ngpu=None,                                    # Optional
    folder=None,                                  # Optional
    secure=True                                   # Optional
)

gan_model.summary()
gan_model.fit(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,           # Optional
    y_test=y_test,           # Optional
    batch_size=32,           # Optional
    epochs=2,                # Optional
    steps=None,              # Optional
    print_every="0.2e",      # Optional
    save_model_every=None,   # Optional
    save_images_every=None,  # Optional
    save_losses_every=10,    # Optional
    enable_tensorboard=False # Optional
)
samples, losses = gan_model.get_training_results(by_epoch=False)

fixed_labels = np.argmax(gan_model.get_fixed_labels(), axis=1)
fig, axs = plot_images(images=samples, labels=fixed_labels, show=False)

test_labels = np.eye(N=10)
test_samples = gan_model.generate(y=test_labels)
test_labels = np.argmax(test_labels, axis=1)
fig, axs = plot_images(images=test_samples, labels=test_labels, show=False)
