from sklearn import datasets
import numpy as np


def generate_dataset():
    np.random.seed(100)
    samples, labels = datasets.make_moons(250, noise=0.10)
    labels = labels.reshape(250, 1)
    return samples, labels


def main():
    samples, labels = generate_dataset()


if __name__ == "__main__":
    main()

