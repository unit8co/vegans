import os
import pickle

import numpy as np
import pandas as pd
import torch.nn as nn

from PIL import Image
from vegans.utils.utils import get_input_dim
from torch.utils.data import DataLoader, Dataset
from vegans.utils.layers import LayerReshape, LayerPrintSize

def preprocess_celeba(root, batch_size, max_loaded_images=5000, **kwargs):
    """ Loader for the CelebA dataset.

    Download from here: https://www.kaggle.com/jessicali9530/celeba-dataset.

    Parameters
    ----------
    root : str
        Path to CelebA root directory.
    batch_size : TYPE
        batch size during training.
    max_loaded_images : TYPE
        Number of examples loaded into memory, before new batch is loaded.
    kwargs
        Other input arguments to torchvision.utils.data.DataLoader

    Returns
    -------
    dataloader
        Class to load in examples dynamically during training.
    """
    class DataSet(Dataset):
        def __init__(self, root, max_loaded_images):
            self.root = root
            self.datapath = root + "CelebA/img_align_celeba/img_align_celeba/"
            self.nr_samples = 202599
            self.max_loaded_images = max_loaded_images
            self.image_shape = (3, 218, 178)
            self.image_names = os.listdir(self.datapath)
            self.current_batch = -1
            self._numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        def __len__(self):
            return self.nr_samples

        def __getitem__(self, index):
            this_batch = index // self.max_loaded_images
            if this_batch != self.current_batch:
                self.current_batch = this_batch
                self.images, self.attributes = self._load_images(start=index)
            index = index % self.max_loaded_images

            return self.images[index], self.attributes[index]

        def _load_images(self, start):
            end = start + self.max_loaded_images
            attributes = pd.read_csv(root + "CelebA/list_attr_celeba.csv").iloc[start:start+end, :]
            attributes = self._transform_targets(targets=attributes)

            batch_image_names = self.image_names[start:end]
            images = np.array([np.array(Image.open(self.datapath+im_name)) for im_name in batch_image_names])
            images = self._transform_data(data=images)
            return images, attributes

        def _transform_targets(self, targets):
            targets = targets.select_dtypes(include=self._numerics).values
            return targets[:, :10]

        def _transform_data(self, data):
            return data.reshape((-1, *self.image_shape)) / 255

    train_dataloader = DataLoader(DataSet(root=root, max_loaded_images=max_loaded_images), batch_size=batch_size, **kwargs)

    return train_dataloader
