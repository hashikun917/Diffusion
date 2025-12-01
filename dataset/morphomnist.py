# https://github.com/biomedia-mira/deepscm/blob/master/deepscm/datasets/morphomnist/__init__.py

import os
import gzip
from typing import Tuple
import struct

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms


MIN_MAX = {
    "thickness": [0.82152224, 6.384839],
    "intensity": [66.48045, 254.93214],
    "slant": [-43.692436, 66.94711],
    "width": [10.000215, 24.999382],
    "image": [0.0, 255.0]
}

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data

def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)


def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path

def load_morphomnist_like(root_dir, train: bool = True, columns=None) \
        -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Args:
        root_dir: path to data directory
        train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
        columns: list of morphometrics to load; by default (``None``) loads the image index and
            all available metrics: area, length, thickness, slant, width, and height
    Returns:
        images, labels, metrics
    """
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = load_idx(images_path)
    labels = load_idx(labels_path)

    if columns is not None and 'index' not in columns:
        usecols = ['index'] + list(columns)
    else:
        usecols = columns
    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col='index')
    return images, labels, metrics


def normalize(data):
    normalized = {}
    for k, v in MIN_MAX.items():
        value = data.images if k == "image" else data.metrics[k]
        # [min,max] -> [0,1]
        normalized[k] = (value - v[0]) / (v[1] - v[0])
        # [0,1] -> [-1,1]
        normalized[k] = 2 * normalized[k] - 1

    return normalized["image"], normalized["thickness"], normalized["intensity"], normalized["slant"], normalized["width"]

def unnormalize(value, name):
    # [-1,1] -> [0,1]
    value = (value + 1) / 2
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

class MorphoMNISTLike(Dataset):
    def __init__(self, root_dir, train: bool = True, columns=None):
        """
        Args:
            root_dir: path to data directory
            train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
                subset (``False``, ``'t10k-*'`` files)
            columns: list of morphometrics to load; by default (``None``) loads the image index and
                all available metrics: area, length, thickness, slant, width, and height
        """
        self.root_dir = root_dir
        self.train = train
        images, labels, metrics_df = load_morphomnist_like(root_dir, train, columns)
        self.images = torch.as_tensor(images)
        self.labels = torch.as_tensor(labels)
        if columns is None:
            columns = metrics_df.columns
        self.metrics = {col: torch.as_tensor(metrics_df[col]) for col in columns}
        self.columns = columns
        assert len(self.images) == len(self.labels) and len(self.images) == len(metrics_df)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.images[idx]
        item['label'] = self.labels[idx]
        return item
    

class MorphoMNISTLikeForClassifier(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True, transform=None, data_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')):
        self.has_valid_set = False
        self.root_dir = data_dir
        self.train = True if split == 'train' else False
        self.transform = transform
        # self.pad = transforms.Pad(padding=2)

        # digit is loaded from labels
        columns = [att for att in attribute_size.keys() if att != 'digit']

        images, labels, metrics_df = load_morphomnist_like(data_dir, self.train, columns)

        self.images = torch.as_tensor(images.copy(), dtype=torch.float32)
        # self.images = self.pad(torch.as_tensor(images.copy(), dtype=torch.float32))
        self.labels = F.one_hot(torch.as_tensor(labels.copy(), dtype=torch.long), num_classes=10)

        if columns is None:
            columns = metrics_df.columns
        self.metrics = {col: torch.as_tensor(metrics_df[col], dtype=torch.float32) for col in columns}
        self.columns = columns
        assert len(self.images) == len(self.labels) and len(self.images) == len(metrics_df)
        if normalize_:
            self.images, self.metrics['thickness'], self.metrics['intensity'], self.metrics['slant'], self.metrics['width'] = normalize(self)

        if "digit" in attribute_size.keys():
            self.metrics["digit"] = self.labels

        self.attrs = torch.cat([self.metrics[attr].unsqueeze(1) if attr != "digit" else self.metrics[attr]
                                for attr in attribute_size.keys()], dim=1)

        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.metrics.items()}

        bins = np.linspace(-1, 1, 10)
        self.bins = {}
        for attr, values in self.metrics.items():
            if attr != "digit":
                data = values.numpy()
                digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.images[idx].unsqueeze(0)
        item['attrs'] = self.attrs[idx]
        if self.transform:
            return self.transform(item["image"], item['attrs'])
        return item['image'], item['attrs']