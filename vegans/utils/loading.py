import os
import pickle
import hashlib
import requests
import subprocess
import torchvision

import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

import vegans.utils.architectures as architectures

class DatasetMetaData():
    def __init__(self, directory, uris, python_hashes):
        self.directory = directory
        self.uris = uris
        self.python_hashes = python_hashes

class DatasetLoader(ABC):
    """
    Class that downloads a dataset and caches it locally.
    Assumes that the file can be downloaded (i.e. publicly available via an URI)

    So far available are:
        - MNIST: Handwritten digits with labels. Can be downloaded via `download=True`.
        - FashionMNIST: Clothes with labels. Can be downloaded via `download=True`.
        - CelebA: Pictures of celebrities with attributes. Must be downloaded from https://www.kaggle.com/jessicali9530/celeba-dataset
                  and moved into `root` folder.
        - CIFAR: Pictures of objects with labels. Must be downloaded from http://www.cs.toronto.edu/~kriz/cifar.html
                  and moved into `root` folder.
    """

    def __init__(self, metadata, root=None):
        self._metadata = metadata
        if root is None:
            self._root = Path(os.path.join(Path.home(), Path('.vegans/datasets/')))
        else:
            self._root = root
        self.path = self._get_path_dataset()

    def load(self):
        """
        Load the dataset in memory, as numpy arrays.
        Downloads the dataset if it is not present already
        """
        if not self._is_already_downloaded():
            self._download_dataset()
        return self._load_from_disk()

    def _is_already_downloaded(self):
        return os.path.exists(self.path)

    @abstractmethod
    def _download_dataset(self):
        """
        Downloads the dataset in the root directory
        """
        os.makedirs(self._root, exist_ok=True)

    def _get_path_dataset(self) -> Path:
        return Path(os.path.join(self._root, self._metadata.directory))

    def _check_dataset_integrity_or_raise(self, path, expected_hash):
        """
        Ensures that the dataset exists and its MD5 checksum matches the expected hash.
        """
        actual_hash = str(subprocess.check_output(["md5sum", path]).split()[0], 'utf-8')
        if actual_hash != expected_hash:
            raise ValueError("Expected hash for {}: {}, got: {}.".format(path, expected_hash, actual_hash))

    @abstractmethod
    def _load_from_disk(self):
        """
        Given a Path to the file and a DataLoaderMetadata object, returns train and test sets as numpy arrays.
        One can assume that the file exists and its MD5 checksum has been verified before this function is called

        Parameters
        ----------
        path_to_file: Path
            A Path object where the dataset is located
        metadata: Metadata
            The dataset's metadata
        """
        pass

    @abstractmethod
    def load_generator(self):
        """ Loads a working generator architecture
        """
        pass

    @abstractmethod
    def load_adversary(self):
        """ Loads a working adversary architecture
        """
        pass

    @abstractmethod
    def load_encoder(self):
        """ Loads a working encoder architecture
        """
        pass

    @abstractmethod
    def load_autoencoder(self):
        """ Loads a working autoencoder architecture
        """
        pass

    @abstractmethod
    def load_decoder(self):
        """ Loads a working generator architecture
        """
        pass


class ExampleLoader(DatasetLoader):

    def _download_dataset(self):
        raise NotImplementedError("No corresponding dataset to this DatasetLoader. Used exclusively to load architectures.")

    def _load_from_disk(self, path_to_file):
        raise NotImplementedError("No corresponding dataset to this DatasetLoader. Used exclusively to load architectures.")

    def load_generator(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_generator(x_dim, z_dim, y_dim=y_dim)

    def load_adversary(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_adversary(x_dim, z_dim, y_dim=y_dim, adv_type=adv_type)

    def load_encoder(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_encoder(x_dim, z_dim, y_dim=y_dim)

    def load_autoencoder(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_autoencoder(x_dim, z_dim, y_dim=y_dim)

    def load_decoder(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_decoder(x_dim, z_dim, y_dim=y_dim)


class MNISTLoader(DatasetLoader):

    def __init__(self, x_dim=(1, 32, 32), z_dim=(64, ), y_dim=(10, ), root=None):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        uris = {
            "data": "",
            "targets": ""
        }
        python_hashes = {
            "data": "9e7c1694ff8fa70086505beb76ee1bda",
            "targets": "06915ca44ac91e0fa65792d391bec292"
        }
        metadata = DatasetMetaData(directory="MNIST", uris=uris, python_hashes=python_hashes)
        super().__init__(metadata=metadata, root=root)

    def _download_dataset(self):
        super()._download_dataset()
        for key, uri in self._metadata.uris.items():
            try:
                request = requests.get(uri)
                with open(self.path, "wb") as f:
                    f.write(request.content)
            except Exception as e:
                raise ValueError("Could not download the dataset. Reason :" + e.__repr__()) from None

    def _load_from_disk(self):
        path = os.path.join(self.path, "mnist_data.pickle")
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._check_dataset_integrity_or_raise(path=path, expected_hash=self._metadata.python_hashes["data"])
            X_train, X_test = data["train"], data["test"]

        path = os.path.join(self.path, "mnist_targets.pickle")
        if self.y_dim is not None:
            with open(os.path.join(self.path, "mnist_targets.pickle"), "rb") as f:
                targets = pickle.load(f)
                self._check_dataset_integrity_or_raise(path=path, expected_hash=self._metadata.python_hashes["targets"])
                y_train, y_test = targets["train"], targets["test"]
        else:
            y_train = None
            y_test = None

        X_train, y_train, X_test, y_test = self._preprocess(X_train, y_train, X_test, y_test)
        if y_train is None:
            return X_train, X_test
        return X_train, y_train, X_test, y_test

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

    def load_generator(self):
        return architectures.load_mnist_generator(z_dim=self.z_dim, y_dim=self.ydim)

    def load_adversary(self, adv_type):
        return architectures.load_mnist_adversary(adv_type=adv_type, y_dim=self.ydim)

    def load_encoder(self):
        return architectures.load_mnist_encoder(x_dim=self.x_dim, z_dim=self.z_dim, y_dim=self.ydim)

    def load_autoencoder(self):
        return architectures.load_mnist_autoencoder(z_dim=self.z_dim, y_dim=self.ydim)

    def load_decoder(self):
        return architectures.load_mnist_decoder(z_dim=self.z_dim, y_dim=self.ydim)

