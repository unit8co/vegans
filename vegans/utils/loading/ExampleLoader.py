import vegans.utils.loading.architectures as architectures

from vegans.utils.loading.DatasetLoader import DatasetLoader

class ExampleLoader(DatasetLoader):

    def __init__(self):
        pass

    def _load_from_disk(self, path_to_file):
        raise NotImplementedError("No corresponding dataset to this DatasetLoader. Used exclusively to load architectures.")

    def load_generator(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_adversary(self, x_dim, y_dim=None, adv_type="Discriminator"):
        return architectures.load_example_adversary(x_dim=x_dim, y_dim=y_dim, adv_type=adv_type)

    def load_encoder(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_autoencoder(self, x_dim, y_dim=None):
        return architectures.load_example_autoencoder(x_dim=x_dim, y_dim=y_dim)

    def load_decoder(self, x_dim, z_dim, y_dim=None):
        return architectures.load_example_decoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)