import numpy as np
import vegans.utils.loading.architectures as architectures

from vegans.utils.loading.MNISTLoader import MNISTLoader
from vegans.utils.loading.DatasetLoader import DatasetLoader, DatasetMetaData

class CIFAR10Loader(MNISTLoader):
    def __init__(self, root=None):
        self.path_data = "cifar10_data.pickle"
        self.path_targets = "cifar10_targets.pickle"
        m5hashes = {
            "data": "40e8e2ca6c43feaa1c7c78a9982b978e",
            "targets": "9a7e604de1826613e860e0bce5a6c1d0"
        }
        metadata = DatasetMetaData(directory="CIFAR10", m5hashes=m5hashes)
        DatasetLoader.__init__(self, metadata=metadata, root=root)

    @staticmethod
    def _preprocess(X_train, y_train, X_test, y_test):
        """ Preprocess mnist by normalizing and padding.
        """
        max_number = X_train.max()
        X_train = X_train / max_number
        X_test = X_test / max_number

        if y_train is not None:
            y_train = np.eye(10)[y_train.reshape(-1)]
            y_test = np.eye(10)[y_test.reshape(-1)]
        return X_train, y_train, X_test, y_test

    def load_generator(self, x_dim=(3, 32, 32), z_dim=64, y_dim=10):
        return architectures.load_mnist_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_adversary(self, x_dim=(3, 32, 32), y_dim=10, adv_type="Discriminator"):
        return architectures.load_mnist_adversary(x_dim=x_dim, y_dim=y_dim, adv_type=adv_type)

    def load_encoder(self, x_dim=(3, 32, 32), z_dim=64, y_dim=10):
        return architectures.load_mnist_encoder(x_dim=self.x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_autoencoder(self, z_dim=64, y_dim=10):
        return architectures.load_mnist_autoencoder(z_dim=z_dim, y_dim=y_dim)

    def load_decoder(self, z_dim=64, y_dim=10):
        return architectures.load_mnist_decoder(z_dim=z_dim, y_dim=y_dim)