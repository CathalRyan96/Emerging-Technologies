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

    # Reshape train_img/ changes it to an array
    inputs = train_img.reshape(60000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    print(train_lbl[0], outputs[0])


def startNeuralNetwork():

    # Start a neural network, building it by layers.
    model = kr.models.Sequential()

    # Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=500, activation='relu', input_dim=784))
    model.add(kr.layers.Dense(units=400, activation='relu'))
    model.add(kr.layers.Dense(units=200, activation='relu'))

    # Add a three neuron output layer.
    model.add(kr.layers.Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    # Read all the imported folders
    train_img = ~np.array(list(train_img[16:])).reshape(
        60000, 28, 28).astype(np.uint8) / 255.0
    train_lbl = np.array(list(train_lbl[8:])).astype(np.uint8)

    # Reshape train_img/ changes it to an array
    inputs = train_img.reshape(60000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    print(train_lbl[0], outputs[0])

    model.fit(inputs, outputs, epochs=10, batch_size=100)
    model.save("data/saveModel.h5")


def Tests():
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        t10k_lbl = f.read()

    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        t10k_img = f.read()

     # Read all the imported folders
    train_img = ~np.array(list(train_img[16:])).reshape(
        60000, 28, 28).astype(np.uint8) / 255.0
    train_lbl = np.array(list(train_lbl[8:])).astype(np.uint8)

    t10k_img = ~np.array(list(t10k_img[16:])).reshape(
        10000, 784).astype(np.uint8) / 255.0
    t10k_lbl = np.array(list(t10k_lbl[8:])).astype(np.uint8)

    # Start a neural network, building it by layers.
    model = kr.models.Sequential()

    # Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=500, activation='relu', input_dim=784))
    model.add(kr.layers.Dense(units=400, activation='relu'))
    model.add(kr.layers.Dense(units=200, activation='relu'))

    # Add a three neuron output layer.
    model.add(kr.layers.Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    print("Tests are now starting...")
    from random import randint

    # performs 10 tests
    for i in range(10):
        print(i, encoder.transform([i]))

    print(encoder.inverse_transform(model.predict(t10k_img)) == t10k_lbl).sum()


read_data()

startNeuralNetwork()

Tests()
