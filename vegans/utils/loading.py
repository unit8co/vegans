import os
import wget
import pickle
import subprocess

import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from zipfile import ZipFile
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from vegans.utils.utils import invert_channel_order

import vegans.utils.architectures as architectures

_SOURCE = "https://vegans-data.s3.eu-west-3.amazonaws.com/"

class DatasetMetaData():
    def __init__(self, directory, m5hashes):
        self.directory = directory
        self.m5hashes = m5hashes

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
        Downloads the dataset if it is not present _is_already_downloaded
        """
        if not self._is_already_downloaded():
            self._download_dataset()
        return self._load_from_disk()

    def _is_already_downloaded(self):
        return os.path.exists(self.path)

    def _download_dataset(self):
        print("Downloading {} to {}...".format(self._metadata.directory, self._get_path_dataset()))
        os.makedirs(self._root, exist_ok=True)
        file_name = self._metadata.directory + ".zip"
        source = os.path.join(_SOURCE, file_name)
        target = os.path.join(self._root, file_name)
        wget.download(source, target)
        with ZipFile(target, 'r') as zipObj:
            zipObj.extractall(self._root)
        os.remove(target)

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
        return architectures.load_mnist_encoder(x_dim=self.x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_autoencoder(self, z_dim=32, y_dim=10):
        return architectures.load_mnist_autoencoder(z_dim=z_dim, y_dim=y_dim)

    def load_decoder(self, z_dim=32, y_dim=10):
        return architectures.load_mnist_decoder(z_dim=z_dim, y_dim=y_dim)


class FashionMNISTLoader(MNISTLoader):
    def __init__(self, root=None):
        self.path_data = "fashionmnist_data.pickle"
        self.path_targets = "fashionmnist_targets.pickle"
        m5hashes = {
            "data": "a25612811c69618cdb9f3111446285f4",
            "targets": "a85af1a3c426f56c52911c7a1cfe5b19"
        }
        metadata = DatasetMetaData(directory="FashionMNIST", m5hashes=m5hashes)
        DatasetLoader.__init__(self, metadata=metadata, root=root)


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


class CIFAR100Loader(CIFAR10Loader):
    def __init__(self, root=None):
        self.path_data = "cifar100_data.pickle"
        self.path_targets = "cifar100_targets.pickle"
        m5hashes = {
            "data": "d0fc36fde6df99d13fc8d9b20a87bd37",
            "targets": "48495792f9c4d719b84b56127d4d725a"
        }
        metadata = DatasetMetaData(directory="CIFAR100", m5hashes=m5hashes)
        DatasetLoader.__init__(self, metadata=metadata, root=root)

    @staticmethod
    def _preprocess(X_train, y_train, X_test, y_test):
        """ Preprocess mnist by normalizing and padding.
        """
        max_number = X_train.max()
        X_train = X_train / max_number
        X_test = X_test / max_number

        if y_train is not None:
            y_train = np.eye(100)[y_train.reshape(-1)]
            y_test = np.eye(100)[y_test.reshape(-1)]
        return X_train, y_train, X_test, y_test


class CelebALoader(DatasetLoader):
    def __init__(self, x_dim=(3, 218, 178), z_dim=(128, ), y_dim=(10, ), root=None, batch_size=32, max_loaded_images=5000, **kwargs):
        """
        Parameters
        ----------
        batch_size : int
            batch size during training.
        max_loaded_images : int
            Number of examples loaded into memory, before new batch is loaded.
        kwargs
            Other input arguments to torchvision.utils.data.DataLoader
        """
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.max_loaded_images = max_loaded_images
        self.kwargs = kwargs
        m5hashes = {
            "targets": "55dfc34188defde688032331b34f9286"
        }
        metadata = DatasetMetaData(directory="CelebA", m5hashes=m5hashes)
        DatasetLoader.__init__(self, metadata=metadata, root=root)

    def _load_from_disk(self):
        class DataSet():
            def __init__(self, root, max_loaded_images, load_attributes):
                self.root = root
                self.datapath = os.path.join(root, "CelebA/images/")
                self.attributepath = os.path.join(root, "CelebA/list_attr_celeba.csv")
                self.nr_samples = 202599
                self.max_loaded_images = max_loaded_images
                self.image_shape = (3, 218, 178)
                self.load_attributes = load_attributes
                try:
                    self.image_names = os.listdir(self.datapath)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "No such file or directory: '{}'. Download from: https://www.kaggle.com/jessicali9530/celeba-dataset."
                        .format(self.datapath)
                    )
                self.current_batch = -1
                self._numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            def __len__(self):
                return self.nr_samples

            def __getitem__(self, index):
                this_batch = index // self.max_loaded_images
                if this_batch != self.current_batch:
                    self.current_batch = this_batch
                    self.images, self.attributes = self._load_data(start=index)
                index = index % self.max_loaded_images

                if self.attributes is None:
                    self.images[index]
                return self.images[index], self.attributes[index]

            def _load_data(self, start):
                end = start + self.max_loaded_images

                if self.load_attributes:
                    attributes = pd.read_csv(self.attributepath).iloc[start:start+end, :]
                    attributes = self._transform_targets(targets=attributes)
                else:
                    attributes = None

                batch_image_names = self.image_names[start:end]
                images = np.array([np.array(Image.open(self.datapath+im_name)) for im_name in batch_image_names])
                images = self._transform_data(data=images)
                return images, attributes

            def _transform_targets(self, targets):
                targets = targets.select_dtypes(include=self._numerics).values
                return targets

            def _transform_data(self, data):
                data = invert_channel_order(images=data)
                return data.reshape((-1, *self.image_shape)) / 255

        self._check_dataset_integrity_or_raise(
            path=os.path.join(self._root, "CelebA/list_attr_celeba.csv"), expected_hash=self._metadata.m5hashes["targets"]
        )
        train_dataloader = DataLoader(DataSet(
                root=self._root, max_loaded_images=self.max_loaded_images, load_attributes=self.y_dim is not None
            ), batch_size=self.batch_size, **self.kwargs
        )

        return train_dataloader

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
