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
import mnist_model

import os


def generate_adversarial(input_image, input_label, model):
    # from https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

    input_image = np.reshape(
        input_image,
        (1, input_image.shape[0], input_image.shape[1], input_image.shape[2]),
    )
    input_label = np.reshape(input_label, (1, input_label.shape[0]))
    input_tensor = tf.convert_to_tensor(input_image)
    label_tensor = tf.convert_to_tensor(input_label)

    prediction = model(input_tensor)
    print(input_label.shape)
    print(prediction.shape)
    loss_object = tf.compat.v1.losses.softmax_cross_entropy(
        input_label, logits=prediction
    )

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        # loss = loss_object(model, input_tensor, input_label)
        gradient = tape.gradient(loss_object, input_tensor)

    # Get the gradients of the loss w.r.t to the input image.
    print("Loss object", loss_object)
    print("Input tensor", np.shape(input_tensor))

    # Get the sign of the gradients to create the perturbation
    print(gradient)
    signed_grad = tf.sign(gradient)
    # return signed_grad


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
        x_train, y_train, x_test, y_test, input_shape = mnist_model.preprocessing(
            x_train, y_train, x_test, y_test
        )
        base_example = x_train[0]
        base_label = y_train[0]

        generate_adversarial(base_example, base_label, model)
    else:
        print("model file does not exist")


if __name__ == "__main__":
    main()
