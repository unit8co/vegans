import vegans.utils.loading.architectures as architectures

from vegans.utils.loading.MNISTLoader import MNISTLoader
from vegans.utils.loading.DatasetLoader import DatasetLoader, DatasetMetaData

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