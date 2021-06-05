import numpy as np
import vegans.utils.loading.architectures as architectures

from PIL import Image
from vegans.utils.loading.CIFAR10Loader import CIFAR10Loader
from vegans.utils.loading.DatasetLoader import DatasetLoader, DatasetMetaData

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