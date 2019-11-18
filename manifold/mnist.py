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

from models import mnist_model
import fgsm
import learning_methods as lm

import os

WIDTH = 28
HEIGHT = 28
EPSILON = .15 # example values: low = .01, high = .15
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
def display(input):
    if np.ndim(input) > 2:
        input = input.reshape((HEIGHT, WIDTH))
    plt.figure()
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
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_org = y_train
        x_train, y_train, x_test, y_test, input_shape = mnist_model.preprocessing(
            x_train, y_train, x_test, y_test
        )
        
        m_x = x_train.reshape((60000, 28*28))
        color_list = ["red", "orange", "yellow", "lime", "green", "cyan", "blue", "purple", "fuchsia", "peru"]
        colors = [color_list[y_org[j]]  for j in range(5000)]
        print(np.shape(colors))
        
        manifold = lm.lle(m_x[:5000])
        print(np.shape(manifold))
        plt.scatter(manifold[:,0], manifold[:,1], c=colors)
        plt.show()

        # Choose example to perturb
        base_example = x_train[0] # Dimensions (HEIGHT, WIDTH, 1)
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
        init_conf = init_preds[0,init_pred]
        print("Initial prediction: " 
            + str(init_pred)
            + " with confidence: " 
            + str(init_conf))
        new_preds = model.predict(adv_x)
        new_pred = np.argmax(new_preds)
        new_conf = new_preds[0, new_pred]
        print("New prediction: "
            + str(new_pred)
            + " with confidence: "
            + str(new_conf))
        
        # Display examples
        display(base_example)
        display(np_pert)
        display(adv_x)
        plt.show()

        # TODO:
            # Learn manifold
            # Find distance of new example to manifold
            # Recalculate distance metric

        # Learn manifold
        

    else:
        print("model file does not exist")


if __name__ == "__main__":
    main()
