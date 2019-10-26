from sklearn import datasets
import numpy as np

import keras
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D, Reshape
from keras.models import Sequential
from keras.utils import to_categorical


def generate_dataset():
    np.random.seed(100)
    samples, labels = datasets.make_moons(250, noise=0.10)
    labels = labels.reshape(250, 1)
    return samples, labels


def generate_moon_model():
    model = Sequential()
    model.add(Conv1D(64, (1), input_shape=(250, 2)))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(32, (1)))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=(5)))

    model.add(Dense(20))
    model.add(Reshape((-1, 2)))
    model.summary()

    model.add(Activation("softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def main():
    samples, labels = generate_dataset()
    samples = np.array([samples])
    one_hot = np.array([to_categorical(labels)])
    model = generate_moon_model()
    model.fit(samples, one_hot, epochs=310)


if __name__ == "__main__":
    main()

