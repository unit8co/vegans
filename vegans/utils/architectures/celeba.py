import os
import pickle

import numpy as np
import pandas as pd
import torch.nn as nn

from PIL import Image
from vegans.utils.utils import get_input_dim
from torch.utils.data import DataLoader, Dataset
from vegans.utils.layers import LayerReshape, LayerPrintSize

def preprocess_celeba(root, batch_size, max_loaded_images=100, **kwargs):
    """ Loader for the CelebA dataset.

    Download from here: https://www.kaggle.com/jessicali9530/celeba-dataset.

    Parameters
    ----------
    root : str
        Path to CelebA root directory
    nr_batches : TYPE
        Number of batches to be constructed / read.

    Returns
    -------
    dataloader
        Class to load in examples dynamically during training.
    """
    # batch_size = 20000
    # nr_samples = 202599

    # for i, start in enumerate(range(0, nr_samples, batch_size)):
    #     print("Batch {} / {}".format(start, nr_samples))
    #     if i+1 <= nr_batches and not os.path.exists(root+"CelebA/batch_{}.pickle".format(start)):
    #         datapath = root + "CelebA/img_align_celeba/img_align_celeba/"
    #         image_names = os.listdir(datapath)
    #         attributes = pd.read_csv(root + "CelebA/list_attr_celeba.csv")
    #         batch_image_names = image_names[start:start+batch_size]
    #         images = np.array([np.array(Image.open(datapath+im_name)) for im_name in batch_image_names])
    #         with open(root+"CelebA/batch_{}.pickle".format(start), "wb") as f:
    #             save_data = {"data": images, "targets": attributes.iloc[start:start+batch_size, :]}
    #             pickle.dump(save_data, f)

    # with open(root+"CelebA/batch_0.pickle", "rb") as f:
    #     load_data = pickle.load(f)
    # X_train = load_data["data"]
    # y_train = load_data["targets"]
    # for i in range(nr_batches-1):
    #     with open(root+"CelebA/batch_{}.pickle".format(i), "rb") as f:
    #         load_data = pickle.load(f)
    #         X_train = np.concatenate((X_train, load_data["data"]), axis=0)
    #         y_train = pd.concat((y_train, load_data["targets"]))

    class DataSet(Dataset):
        def __init__(self, root, max_loaded_images):
            self.root = root
            self.datapath = root + "CelebA/img_align_celeba/img_align_celeba/"
            self.nr_samples = 202599
            self.nr_samples = 500
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
            attributes = attributes.select_dtypes(include=self._numerics).values
            attributes = self._transform_targets(targets=attributes)

            batch_image_names = self.image_names[start:end]
            images = np.array([np.array(Image.open(self.datapath+im_name)) for im_name in batch_image_names])
            images = self._transform_data(data=images)
            return images, attributes

        def _transform_targets(self, targets):
            return targets

        def _transform_data(self, data):
            return data.reshape((-1, *self.image_shape))

    train_dataloader = DataLoader(DataSet(root=root, max_loaded_images=max_loaded_images), batch_size=batch_size, **kwargs)

    return train_dataloader
