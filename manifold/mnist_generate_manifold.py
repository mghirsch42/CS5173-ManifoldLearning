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
import argparse

import mnist_model
import fgsm
import learning_methods as lm
import mnist_model
import mnist_tools

import os

'''
Loads pretrained MNIST model
Learns manifold on data
'''

WIDTH = 28      # Width of MNIST data
HEIGHT = 28     # Height of MNIST data
PATH_TO_MODEL = "models/mnist_model.h5"     # Path to trained network model
NUM_EX = 10             # Number of examples of data to use to generate manifolds
LEARNING_METHOD = "lle"    # Manifold learning method to use


def main():
    if os.path.exists(PATH_TO_MODEL):
        model = load_model(PATH_TO_MODEL)
        model = mnist_tools.convert_to_model(model)
        model.trainable = True
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        
        # Colors for manifold visualization
        color_list = mnist_tools.color_list
        colors = [color_list[y_train[j]] for j in range(len(y_train))]

        # Preprocess examples
        x_train, y_train, x_test, y_test, input_shape = mnist_model.preprocessing(
            x_train, y_train, x_test, y_test
        )

        # Manifold learning methods need flattened data
        m_x = x_train.reshape((len(x_train), WIDTH * HEIGHT))

        # Generate manifold
        if LEARNING_METHOD == "lle":
            embedding, manifold = lm.lle(m_x[:NUM_EX])
        elif LEARNING_METHOD == "isomap":
            embedding, manifold = lm.isomap(m_x[:NUM_EX])
        elif LEARNING_METHOD == "laplacian":
            embedding, manifold = lm.laplacian(m_x[:NUM_EX])
        elif LEARNING_METHOD == "tsne":
            embedding, manifold = lm.tsne(m_x[:NUM_EX])
        else:
            print("Invalid learning method")
            exit(0)
         

        # Save - change file names / comment as needed
        np.save("mnist-{}-{}-2d".format(LEARNING_METHOD, NUM_EX), manifold)
        pickle.dump(embedding, open("mnist-{}-{}-2d.pkl".format(LEARNING_METHOD, NUM_EX), "wb"))
        

        # Plot
        plt.scatter(manifold[:, 0], manifold[:, 1], c=colors[:NUM_EX])
        plt.show()


    else:
        print("Model file does not exist")


if __name__ == "__main__":
    main()
