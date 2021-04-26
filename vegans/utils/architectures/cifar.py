import os
import pickle

import numpy as np

def preprocess_cifar(root, normalize=True, pad=None):
    """Load the mnist data from root.

    Download from here: http://www.cs.toronto.edu/~kriz/cifar.html.

    Parameters
    ----------
    root : str
        Path to cifar10 root folder.
    normalize : bool, optional
        If True, data will be scaled to the interval [0, 1]
    pad : None, optional
        Integer indicating the padding applied to each side of the input images.

    Returns
    -------
    numpy.array
        train and test data as well as labels.
    """
    root = os.path.join(root, "CIFAR10")

    if os.path.exists(os.path.join(root, "train_data.pickle")):
        with open(os.path.join(root, "train_data.pickle"), "rb") as f:
            train_data = pickle.load(f)
            X_train = train_data["data"]
            y_train = train_data["targets"]
    else:
        try:
            train_files = [os.path.join(root, f) for f in os.listdir(root) if "data" in f]
        except FileNotFoundError:
            raise FileNotFoundError(
                "No such file or directory: '{}'. Download from: http://www.cs.toronto.edu/~kriz/cifar.html."
                .format(root)
            )
        with open(train_files[0], "rb") as f:
            train_data = pickle.load(f, encoding='bytes')
            X_train = train_data[b"data"].reshape((-1, 3, 32, 32))
            y_train = train_data[b"labels"]
        for train_file in train_files[1:]:
            with open(train_file, "rb") as f:
                train_data = pickle.load(f, encoding='bytes')
                X_train = np.concatenate((X_train, train_data[b"data"].reshape((-1, 3, 32, 32))))
                y_train = np.concatenate((y_train, train_data[b"labels"]))

        with open(os.path.join(root, "train_data.pickle"), "wb") as f:
            pickle.dump({"data": X_train, "targets": y_train}, f)

    with open(os.path.join(root, "test_batch"), "rb") as f:
        test_data = pickle.load(f, encoding='bytes')
        test_images = test_data[b"data"]
        red_channel = test_images[:, :1024].reshape((-1, 32, 32))
        green_channel = test_images[:, 1024:2048].reshape((-1, 32, 32))
        blue_channel = test_images[:, 2048:3072].reshape((-1, 32, 32))
        X_test = np.stack((red_channel, green_channel, blue_channel), axis=1)
        y_test = np.array(test_data[b"labels"])

    if normalize:
        max_number = X_train.max()
        X_train = X_train / max_number
        X_test = X_test / max_number

    if pad:
        X_train = np.pad(X_train, [(0, 0), (pad, pad), (pad, pad)], mode='constant')
        X_test = np.pad(X_test, [(0, 0), (pad, pad), (pad, pad)], mode='constant')

    return X_train, y_train, X_test, y_test