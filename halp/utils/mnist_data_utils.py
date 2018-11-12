import gzip
import os
from os import path
import numpy as np
import torch

import sys
if sys.version_info.major < 3:
    import urllib
else:
    import urllib.request as request


DATASET_DIR = 'datasets/'

MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def download_file(url, local_path):
    dir_path = path.dirname(local_path)
    if not path.exists(dir_path):
        print("Creating the directory '%s' ..." % dir_path)
        os.makedirs(dir_path)

    print("Downloading from '%s' ..." % url)
    if sys.version_info.major < 3:
        urllib.URLopener().retrieve(url, local_path)
    else:
        request.urlretrieve(url, local_path)


def download_mnist(local_path):
    url_root = "http://yann.lecun.com/exdb/mnist/"
    for f_name in MNIST_FILES:
        f_path = os.path.join(local_path, f_name)
        if not path.exists(f_path):
            download_file(url_root + f_name, f_path)


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def load_mnist(ntrain=60000, ntest=10000, onehot=True):
    data_dir = os.path.join(DATASET_DIR, 'mnist/')
    if not path.exists(data_dir):
        download_mnist(data_dir)
    else:
        # check all files
        checks = [path.exists(os.path.join(data_dir, f)) for f in MNIST_FILES]
        if not np.all(checks):
            download_mnist(data_dir)

    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trY = loaded[8:].reshape((60000))

    with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teY = loaded[8:].reshape((10000))

    trX /= 255.
    teX /= 255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY

def get_mnist_data_loader(onehot=False, debug_test=False, batch_size=1):
    X_train, X_val, Y_train, Y_val = load_mnist(onehot=False)
    if debug_test:
        debug_data_size = 3
        X_train = X_train[0:debug_data_size]
        X_val = X_val[0:debug_data_size]
        Y_train = Y_train[0:debug_data_size]
        Y_val = Y_val[0:debug_data_size]
    X_train, X_val = torch.FloatTensor(X_train), torch.FloatTensor(X_val)
    Y_train, Y_val = torch.LongTensor(Y_train), torch.LongTensor(Y_val)
    train_data = \
        torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False)
    val_data = \
        torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False)
    input_shape = (batch_size,) + X_train.shape[1:]
    return train_loader, val_loader, input_shape, X_train.shape[0]