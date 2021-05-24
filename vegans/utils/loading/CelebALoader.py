import os

import numpy as np
import pandas as pd
import vegans.utils.loading.architectures as architectures

from PIL import Image
from torch.utils.data import DataLoader
from vegans.utils.utils import invert_channel_order
from vegans.utils.loading.DatasetLoader import DatasetLoader, DatasetMetaData

class CelebALoader(DatasetLoader):
    def __init__(self, root=None, batch_size=32, max_loaded_images=5000, crop_size=128, output_shape=64, verbose=False, **kwargs):
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
        self.batch_size = batch_size
        self.max_loaded_images = max_loaded_images
        self.crop_size = crop_size
        self.output_shape = output_shape
        self.verbose = verbose
        self.kwargs = kwargs
        m5hashes = {
            "targets": "55dfc34188defde688032331b34f9286"
        }
        metadata = DatasetMetaData(directory="CelebA", m5hashes=m5hashes)
        DatasetLoader.__init__(self, metadata=metadata, root=root)

    def _load_from_disk(self):
        class DataSet():
            def __init__(self, root, max_loaded_images, crop_size, output_shape, verbose):
                self.root = root
                self.datapath = os.path.join(root, "CelebA/images/")
                self.attributepath = os.path.join(root, "CelebA/list_attr_celeba.csv")
                self.nr_samples = 202599
                self.max_loaded_images = max_loaded_images
                self.verbose = verbose
                self.original_shape = (3, 218, 178)
                self.crop_size = crop_size
                self.output_shape = output_shape
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
                    if self.verbose:
                        print("Loaded image batch {} / {}.".format(this_batch, len(self)//self.max_loaded_images))

                index = index % self.max_loaded_images

                if self.attributes is None:
                    self.images[index]
                return self.images[index], self.attributes[index]

            def _load_data(self, start):
                end = start + self.max_loaded_images

                attributes = pd.read_csv(self.attributepath).iloc[start:start+end, :]
                attributes = self._transform_targets(targets=attributes)

                batch_image_names = self.image_names[start:end]
                images = [self._transform_image(Image.open(self.datapath+im_name)) for im_name in batch_image_names]
                # images = self._transform_data(data=images)
                return images, attributes

            def _transform_targets(self, targets):
                targets = targets.select_dtypes(include=self._numerics).values
                return targets

            def _transform_data(self, data):
                for i, image in enumerate(data):
                    left_x = (image.size[0] - self.crop_size) // 2
                    upper_y = (image.size[1] - self.crop_size) // 2
                    image = image.crop([left_x, upper_y, left_x + self.crop_size, upper_y + self.crop_size])
                    image = image.resize((self.output_shape, self.output_shape), Image.BILINEAR)
                    data[i] = np.array(image)
                data = invert_channel_order(images=np.stack(data, axis=0))
                max_value = np.max(data)
                return data / max_value

            def _transform_image(self, image):
                left_x = (image.size[0] - self.crop_size) // 2
                upper_y = (image.size[1] - self.crop_size) // 2
                image = image.crop([left_x, upper_y, left_x + self.crop_size, upper_y + self.crop_size])
                image = image.resize((self.output_shape, self.output_shape), Image.BILINEAR)
                image =  np.array([np.array(image)])
                image = invert_channel_order(images=image)[0, :]
                return image / 255

        self._check_dataset_integrity_or_raise(
            path=os.path.join(self._root, "CelebA/list_attr_celeba.csv"), expected_hash=self._metadata.m5hashes["targets"]
        )
        train_dataloader = DataLoader(
            DataSet(
                root=self._root, max_loaded_images=self.max_loaded_images,
                crop_size=self.crop_size, output_shape=self.output_shape,
                verbose=self.verbose
            ),
            batch_size=self.batch_size, **self.kwargs
        )
        return train_dataloader

    def load_generator(self, x_dim=None, z_dim=(16, 4, 4), y_dim=40):
        if x_dim is None:
            x_dim = (3, self.output_shape, self.output_shape)
        return architectures.load_celeba_generator(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_adversary(self, x_dim=None, y_dim=40, adv_type="Discriminator"):
        if x_dim is None:
            x_dim = (3, self.output_shape, self.output_shape)
        return architectures.load_celeba_adversary(x_dim=x_dim, y_dim=y_dim, adv_type=adv_type)

    def load_encoder(self, x_dim=None, z_dim=(16, 4, 4), y_dim=40):
        if x_dim is None:
            x_dim = (3, self.output_shape, self.output_shape)
        return architectures.load_celeba_encoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)

    def load_autoencoder(self, x_dim=None, y_dim=40):
        if x_dim is None:
            x_dim = (3, self.output_shape, self.output_shape)
        raise NotImplementedError("Autoencoder architecture not defined for `CelebALoader.`")

    def load_decoder(self, x_dim=None, z_dim=(16, 4, 4), y_dim=40):
        if x_dim is None:
            x_dim = (3, self.output_shape, self.output_shape)
        return architectures.load_celeba_decoder(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
