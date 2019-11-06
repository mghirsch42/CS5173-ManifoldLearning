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
import mnist

import os


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
        # print(input_tensor)
        # print(input_label)
        # print(prediction)
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
    if os.path.exists("models/mnist_model.h5"):
        model = load_model("models/mnist_model.h5")
        model = convert_to_model(model)
        model.trainable = True
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, y_train, x_test, y_test, input_shape = mnist.preprocessing(
            x_train, y_train, x_test, y_test
        )
        base_example = x_train[0]
        base_label = y_train[0]

        generate_adversarial(base_example, base_label, model)


if __name__ == "__main__":
    main()
