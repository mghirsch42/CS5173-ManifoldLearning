import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras import datasets
import math
from PIL import Image

import mnist_manifold
import mnist_model

import warnings
warnings.filterwarnings("ignore")

'''
Load MNIST model, manifold model, and transformed adversarial examples
Loop through adversarial examples, find prediction, adjust prediction 
based on manifold and norms
Report statistics
'''


PATH_TO_MODEL = "models/mnist_model.h5"
PATH_TO_DATA = "datasets/test"


def l1(point, manifold):
    l1_distances = [
        abs(point[0] - man_point[0]) + abs(point[1] - man_point[1])
        for man_point in manifold
    ]
    return np.mean(l1_distances)


def l2(point, manifold):
    l2_distances = [
        np.sqrt(
            (point[0] - man_point[0]) * (point[0] - man_point[0])
            + (point[1] - man_point[1]) * (point[1] - man_point[1])
        )
        for man_point in manifold
    ]
    return np.mean(l2_distances)


def fractional_norm(point, manifold, denominator):

    frac_distances = [
        (
            (point[0] - man_point[0]) ** (1 / denominator)
            + (point[1] - man_point[1]) ** (1 / denominator)
        )
        ** denominator
        for man_point in manifold
    ]

    return np.nanmean(frac_distances)


def similarity(point, manifold, method, denominator=1):
    distance = 0.0
    if method == "L1":
        distance = l1(point, manifold)
    elif method == "L2":
        distance = l2(point, manifold)
    elif method == "fraction":
        distance = fractional_norm(point, manifold, denominator)
    return distance


def mnist_preprocessing():
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
    x_test = x_test[:500]
    y_test = y_test[:500]

    return x_test, y_test, colors


def setup_model():
    model = load_model(PATH_TO_MODEL)
    model = mnist_manifold.convert_to_model(model)
    model.trainable = True

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    return model


def pred_confidence(model, example, label):
    orig_confs = model.predict(example)
    pred = np.argmax(orig_confs)
    pred_conf = orig_confs[0, pred]
    print(label + " prediction: " + str(pred) + " with confidence: " + str(pred_conf))
    return pred


def main():
    # load data, manifold, and embedding
    x_test, y_test, colors = mnist_preprocessing()

    manifold = np.load(
        os.path.join(os.path.dirname(__file__), "mnist-ptsne-20000-2d.npy")
    )
    embedding = pickle.load(
        open(os.path.join(os.path.dirname(__file__), "mnist-ptsne-20000-2d.pkl"), "rb")
    )

    # for later statistical analysis
    net_incorrect = 0
    net_correct = 0
    net_incorrect_sim_incorrect = 0
    net_incorrect_sim_correct = 0
    net_correct_sim_incorrect = 0
    net_correct_sim_correct = 0

    if os.path.exists(PATH_TO_MODEL):
        model = setup_model()

        # how good the model is without adversarial examples
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Original test loss:", score[0])
        print("Original test accuracy:", score[1])

        adversarials = []
        for i in range(0, 500):
            adversarial = np.load(os.path.join(PATH_TO_DATA, str(i) + ".npy"))
            # adversarial = np.random.randint(2, size=(1, 28, 28, 1))
            adversarials.append(adversarial)

            adv_pred = pred_confidence(model, adversarial, "Adversarial")
            orig_pred = pred_confidence(model, np.array([x_test[i]]), "Original")

            print("Ground truth: " + str(np.argmax(y_test[i])))

            adv_mx = embedding.transform(
                adversarial.reshape(1, adversarial.shape[1] * adversarial.shape[2])
            )

            int_labels = np.argmax(y_test, axis=1)
            sims = []

            # calculate similarity per class
            for k in range(0, 10):
                k_list = np.array(
                    list(filter(lambda x: int_labels[x] == k, range(len(int_labels))))
                )

                sim = similarity(adv_mx[0], manifold[k_list], "fraction", 2)
                sims.append(sim)
                print(str(k) + ": " + str(sim))

            min_sim = sims.index(min(sims))

            # incorrect prediction from net
            if adv_pred != np.argmax(y_test[i]):
                mnist_manifold.display(adversarial, "Adversarial Example")
                plt.show()

                plt.figure()
                plt.scatter(manifold[:, 0], manifold[:, 1], c=colors[:20000], s=25)
                print(adv_mx)
                plt.scatter(adv_mx[0][0], adv_mx[0][1], c="k", s=40)
                plt.show()

                net_incorrect += 1
                # attempted correction is correct
                if min_sim == np.argmax(y_test[i]):
                    net_incorrect_sim_correct += 1
                # attempted correction is incorrect
                else:
                    net_incorrect_sim_incorrect += 1

            else:
                net_correct += 1
                # attempted correction is correct
                if min_sim == np.argmax(y_test[i]):
                    net_correct_sim_correct += 1
                # attempted correction is incorrect
                else:
                    net_correct_sim_incorrect += 1

    else:
        print("Model file does not exist")

    score = model.evaluate(
        np.array(adversarials).reshape(500, 28, 28, 1), y_test, verbose=0
    )
    print("Adversarial test loss:", score[0])
    print("Adversarial test accuracy:", score[1])

    print(
        "Incorrect labels corrected: " + str(net_incorrect_sim_correct / net_incorrect)
    )
    print(
        "Incorrect labels not corrected: "
        + str(net_incorrect_sim_incorrect / net_incorrect)
    )
    print()
    print("Correct labels unchanged: " + str(net_correct_sim_correct / net_correct))
    print("Correct labels changed: " + str(net_correct_sim_incorrect / net_correct))


if __name__ == "__main__":
    main()
