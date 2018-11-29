# Import Libraries Needed
import gzip
import numpy as np
import keras as kr


def read_data():
    # Start a neural network
    model = kr.models.Sequential()

    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    print("File Type: ", type(train_img))
    print(train_img[0:4])


read_data()
