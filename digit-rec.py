# Import Libraries Needed
import gzip
import numpy as np
import keras as kr
import sklearn.preprocessing as pre


def read_data():

    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        t10k_lbl = f.read()

    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        t10k_img = f.read()

    print("Testing to see File type: ", type(train_img))
    print("Testing to see if first 4 bytes are printed out: ", train_img[0:4])

    # Read all the imported folders
    train_img = ~np.array(list(train_img[16:])).reshape(
        60000, 28, 28).astype(np.uint8) / 255.0
    train_lbl = np.array(list(train_lbl[8:])).astype(np.uint8)

    t10k_img = ~np.array(list(t10k_img[16:])).reshape(
        10000, 28, 28).astype(np.uint8) / 255.0
    t10k_lbl = np.array(list(t10k_lbl[8:])).astype(np.uint8)

    # Reshape train_img
    inputs = train_img.reshape(60000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    print(train_lbl[0], outputs[0])


read_data()
