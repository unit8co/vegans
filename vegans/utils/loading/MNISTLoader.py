import os
import pickle

import numpy as np
import vegans.utils.loading.architectures as architectures

from vegans.utils.loading.DatasetLoader import DatasetLoader, DatasetMetaData

class MNISTLoader(DatasetLoader):

    def __init__(self, root=None):
        self.path_data = "mnist_data.pickle"
        self.path_targets = "mnist_targets.pickle"
        m5hashes = {
            "data": "9e7c1694ff8fa70086505beb76ee1bda",
            "targets": "06915ca44ac91e0fa65792d391bec292"
        }
        metadata = DatasetMetaData(directory="MNIST", m5hashes=m5hashes)
        super().__init__(metadata=metadata, root=root)

    def _load_from_disk(self):
        X_train, X_test = self._load_from_path(
            path=os.path.join(self.path, self.path_data), m5hash=self._metadata.m5hashes["data"]
        )
        y_train, y_test = self._load_from_path(
            path=os.path.join(self.path, self.path_targets), m5hash=self._metadata.m5hashes["targets"]
        )

        X_train, y_train, X_test, y_test = self._preprocess(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test

    def _load_from_path(self, path, m5hash):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._check_dataset_integrity_or_raise(path=path, expected_hash=m5hash)
            train, test = data["train"], data["test"]
        return train, test

    @staticmethod
    def _preprocess(X_train, y_train, X_test, y_test):
        """ Preprocess mnist by normalizing and padding.
        """
        max_number = X_train.max()
        X_train = X_train / max_number
        X_train = np.pad(X_train, [(0, 0), (2, 2), (2, 2)], mode='constant').reshape(60000, 1, 32, 32)

        X_test = X_test / max_number
        X_test = np.pad(X_test, [(0, 0), (2, 2), (2, 2)], mode='constant').reshape(10000, 1, 32, 32)

        if y_train is not None:
            y_train = np.eye(10)[y_train.reshape(-1)]
            y_test = np.eye(10)[y_test.reshape(-1)]
        return X_train, y_train, X_test, y_test

    def load_generator(self, x_dim=(1, 32, 32), z_dim=32, y_dim=10):
        return architectures.load_mnist_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_adversary(self, x_dim=(1, 32, 32), y_dim=10, adv_type="Discriminator"):
        return architectures.load_mnist_adversary(x_dim=x_dim, y_dim=y_dim, adv_type=adv_type)

    def load_encoder(self, x_dim=(1, 32, 32), z_dim=32, y_dim=10):
        return architectures.load_mnist_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_autoencoder(self, x_dim=(1, 32, 32), y_dim=10):
        return architectures.load_mnist_autoencoder(x_dim=x_dim, y_dim=y_dim)

    def load_decoder(self, x_dim=(1, 32, 32), z_dim=32, y_dim=10):
        return architectures.load_mnist_decoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)