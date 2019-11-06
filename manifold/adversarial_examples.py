from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import moons

import os


def preprocessing(x_train, y_train):

    batch_size = 128
    num_classes = 10
    epochs = 2
    # input image dimensions
    img_rows, img_cols = 28, 28

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_train /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    return x_train, y_train


# from https://keras.io/examples/mnist_cnn/
def generate_model():
    batch_size = 128
    num_classes = 10
    epochs = 2

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        # steps_per_epoch=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    # validation_steps=batch_size)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return model


def generate_adversarial(input_image, input_label, model):
    # from https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
    # input_tensor = tf.convert_to_tensor(input_label)
    input_image = np.reshape(input_image, (1, 28, 28, 1))
    input_tensor = tf.Variable(input_image)
    label_tensor = tf.convert_to_tensor(input_label)

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
        print(input_tensor)
        print(input_label)
        print(prediction)
        loss = loss_object(input_label, prediction)
        print(loss)

    # Get the gradients of the loss w.r.t to the input image.
    print("Loss object", loss_object)
    print("Input tensor", np.shape(input_tensor))
    gradient = tape.gradient(loss, input_tensor)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def convert_to_model(seq_model):
    # From https://github.com/keras-team/keras/issues/10386
    input_layer = keras.layers.Input(batch_shape=seq_model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seq_model.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)
    funcmodel = keras.models.Model([input_layer], [prev_layer])

    return funcmodel


def main():
    if os.path.exists("mnist_model.h5"):
        model = load_model("mnist_model.h5")
        model = convert_to_model(model)
        model.trainable = True
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, y_train = preprocessing(x_train, y_train)
        base_example = x_train[0]
        base_label = y_train[0]

        generate_adversarial(base_example, base_label, model)

    else:
        exit(0)
        model = generate_model()
        model.save("mnist_model.h5")


if __name__ == "__main__":
    main()
