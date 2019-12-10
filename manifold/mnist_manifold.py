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
import mnist_tools

import os

'''
Loads  pretrained MNIST model
Loads manifold from file
Creates adversarial image
Tests adversarial image on model
Displays results
'''

WIDTH = 28
HEIGHT = 28
EPSILON = 0.15  # example values: low = .01, high = .15
PATH_TO_MODEL = "models/mnist_model.h5"
MANIFOLD_FILE = "mnist-lle-10-2d"
MANIFOLD_SIZE = 10  # The number of points in the manifold; used for visalization colors


# Reshapes the input to the correct dimensions,
# creates a new figure and displays the input
def display(input, title):
    if np.ndim(input) > 2:
        input = input.reshape((HEIGHT, WIDTH))
    plt.figure()
    plt.title(title)
    plt.imshow(input)


def main():
    if os.path.exists(PATH_TO_MODEL):
        model = load_model(PATH_TO_MODEL)
        model = mnist_tools.convert_to_model(model)
        model.trainable = True
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        
        # Colors to display the manifold
        color_list = mnist_tools.color_list
        colors = [color_list[y_train[j]] for j in range(len(y_train))]
        x_train, y_train, x_test, y_test, input_shape = mnist_model.preprocessing(
            x_train, y_train, x_test, y_test
        )

        # Load manifold
        manifold = np.load(MANIFOLD_FILE + ".npy")
        embedding = pickle.load(open(MANIFOLD_FILE + ".pkl", "rb"))

        # Choose example to perturb
        base_example = x_train[0] 
        base_label = y_train[0]

        # Generate adversarial image
        adv_x = fgsm.generate_adversarial(base_example, base_label, EPSILON, model)
       
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
        plt.scatter(manifold[:, 0], manifold[:, 1], c=colors[:MANIFOLD_SIZE])
        adv_mx = embedding.transform(adv_x.reshape(1, HEIGHT*WIDTH))
        plt.scatter(adv_mx[0][0], adv_mx[0][1], c="k", s=75)

        # Display examples
        display(base_example, "Original Example")
        display(adv_x, "Adversarial Example")
        plt.show()

    else:
        print("Model file does not exist")


if __name__ == "__main__":
    main()
