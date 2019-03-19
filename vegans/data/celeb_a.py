"""
Modification of https://github.com/carpedm20/BEGAN-pytorch/blob/master/download.py
"""
import os
import zipfile

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from .utils import download_file_from_google_drive, prepare_data_dir


def _download_celeb_a(base_path):
    data_path = os.path.join(base_path, 'CelebA')
    images_path = os.path.join(data_path, 'images')
    if os.path.exists(data_path):
        print('[!] Found Celeb-A - skip')
        return

    filename, drive_id= "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(base_path, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(base_path)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    os.rename(os.path.join(base_path, "img_align_celeba"), images_path)
    os.remove(save_path)


# check, if file exists, make link
def _check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)
        os.symlink(rel_link, link_file)


def _add_splits(base_path):
    data_path = os.path.join(base_path, 'CelebA')
    images_path = os.path.join(data_path, 'images')
    train_dir = os.path.join(data_path, 'splits', 'train')
    valid_dir = os.path.join(data_path, 'splits', 'valid')
    test_dir = os.path.join(data_path, 'splits', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # these constants based on the standard CelebA splits
    num_examples = 202599
    train_stop = 162770
    valid_stop = 182637

    for i in range(0, train_stop):
        basename = "{:06d}.jpg".format(i+1)
        _check_link(images_path, basename, train_dir)
    for i in range(train_stop, valid_stop):
        basename = "{:06d}.jpg".format(i+1)
        _check_link(images_path, basename, valid_dir)
    for i in range(valid_stop, num_examples):
        basename = "{:06d}.jpg".format(i+1)
        _check_link(images_path, basename, test_dir)


def get_celeb_a(path):
    prepare_data_dir(path)
    _download_celeb_a(path)
    _add_splits(path)


def celeb_a_data_loader(root, split, batch_size, scale_size, num_workers=2, shuffle=True):
    # make sure the data exists and get it if not
    get_celeb_a(root)

    image_root = os.path.join(root, 'splits', split)

    dataset = ImageFolder(root=image_root, transform=transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Scale(scale_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader
