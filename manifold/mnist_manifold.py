from __future__ import print_function
import keras
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle

import mnist_model
import fgsm
import learning_methods as lm
import mnist_model

import os

#########
## Loads  pretrained MNIST model
## Learns manifold on data
## Creates adversarial images
## Tests adversarial images on model
#########

WIDTH = 28
HEIGHT = 28
EPSILON = 0.15  # example values: low = .01, high = .15
PATH_TO_MODEL = "models/mnist_model.h5"


def convert_to_model(seq_model):
    # From https://github.com/keras-team/keras/issues/10386
    input_layer = keras.layers.Input(batch_shape=seq_model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seq_model.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)
    funcmodel = keras.models.Model([input_layer], [prev_layer])

    return funcmodel


# Reshapes the input to the correct dimensions,
# creates a new figure and displays the input
def display(input, title):
    if np.ndim(input) > 2:
        input = input.reshape((HEIGHT, WIDTH))
    plt.figure()
    plt.title(title)
    plt.imshow(input)


# Returns 4D np array (1, HEIGHT, WIDTH, 1)
def tensor_to_numpy(t):
    sess = K.get_session()
    t_np = sess.run(t)

    # Get rid of the extra dimension
    t_np = t_np.reshape(1, HEIGHT, WIDTH, 1)
    return t_np


def main():
    if os.path.exists(PATH_TO_MODEL):
        model = load_model(PATH_TO_MODEL)
        model = convert_to_model(model)
        model.trainable = True
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        color_list = [
            "red",
            "orange",
            "yellow",
            "lime",
            "green",
            "cyan",
            "blue",
            "purple",
            "fuchsia",
            "peru",
        ]
        colors = [color_list[y_train[j]] for j in range(len(y_train))]
        x_train, y_train, x_test, y_test, input_shape = mnist_model.preprocessing(
            x_train, y_train, x_test, y_test
        )

        m_x = x_train.reshape((len(x_train), WIDTH * HEIGHT))
        # Train
        # embedding, manifold = lm.laplacian(m_x[:20000])
        # Save - change file names / comment as needed
        # np.save("mnist-laplacian-20000-2d", manifold)
        # pickle.dump(embedding, open("mnist-laplacian-20000-2d.pkl", "wb"))
        
        # Plot
        # plt.scatter(manifold[:, 0], manifold[:, 1], c=colors[:20000])
        # plt.show()
        # exit(0)

        # Load
        manifold = np.load("mnist-laplacian-20000-2d.npy")
        embedding = pickle.load(open("mnist-laplacian-20000-2d.pkl", "rb"))

        # Choose example to perturb
        base_example = x_train[0]  # Dimensions (HEIGHT, WIDTH, 1)
        base_label = y_train[0]

        # Create perturbation values
        perturbations = fgsm.generate_adversarial(base_example, base_label, model)
        np_pert = tensor_to_numpy(perturbations)

        # Create adversarial example
        adv_x = base_example + (EPSILON) * perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        adv_x = tensor_to_numpy(adv_x)

        # Print predictions
        init_preds = model.predict(base_example.reshape(1, HEIGHT, WIDTH, 1))
        init_pred = np.argmax(init_preds)
        init_conf = init_preds[0, init_pred]
        print(
            "Initial prediction: "
            + str(init_pred)
            + " with confidence: "
            + str(init_conf)
        )
        new_preds = model.predict(adv_x)
        new_pred = np.argmax(new_preds)
        new_conf = new_preds[0, new_pred]
        print("New prediction: " + str(new_pred) + " with confidence: " + str(new_conf))

        # Plot manifold with adv example
        plt.figure()
        plt.scatter(manifold[:, 0], manifold[:, 1], c=colors[:20000])
        adv_mx = embedding.transform(adv_x.reshape(1, HEIGHT*WIDTH))
        print(adv_mx)
        plt.scatter(adv_mx[0][0], adv_mx[0][1], c="k", s=75)
        plt.show()

        # Display examples
        display(base_example, "Original Example")
        display(np_pert, "Perturbation")
        display(adv_x, "Adversarial Example")
        plt.show()

        # TODO:
        # Find distance of new example to manifold
        # Recalculate distance metric


    else:
        print("model file does not exist")


if __name__ == "__main__":
    main()
