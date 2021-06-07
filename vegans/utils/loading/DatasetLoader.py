import os
import wget
import subprocess

from pathlib import Path
from zipfile import ZipFile
from abc import ABC, abstractmethod

_SOURCE = "https://vegans-data.s3.eu-west-3.amazonaws.com/"
_DEFAULT_ROOT = '.vegans/datasets/'

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
            self._root = Path.home() / _DEFAULT_ROOT
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
        try: # Linux
            actual_hash = str(subprocess.check_output(["md5sum", path]).split()[0], 'utf-8')
        except FileNotFoundError: # Mac
            actual_hash = str(subprocess.check_output(["md5", path]).split()[-1], 'utf-8')

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