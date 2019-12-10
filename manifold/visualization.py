from matplotlib import pyplot as plt
import numpy as np
from keras import datasets
import pickle

import mnist_tools

'''
Used to create visualizations of manifolds and adversarial examples
'''

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

color_list = mnist_tools.color_list
colors = [color_list[y_train[j]] for j in range(len(y_train))]


def load_manifold(filename):
    manifold = np.load(filename)
    return manifold

def load_embedding(filename):
    embedding = pickle.load(open(filename, "rb"))
    return embedding

def plot_manifold(manifold, method, show=True, save=True, fname="", adv_x=None):
    fig, ax = plt.subplots()
    ax.scatter(manifold[:, 0], manifold[:, 1], 
                c=colors[:len(manifold)],
                s=5)
    if adv_x is not None:
        for x in adv_x:
            print(x)
            ax.scatter(x[0], x[1], c="black", s=50)
    # ax.set_ylim([-.0006, .0006]) # Ranges for SE Laplacian
    # ax.set_xlim([-.0006, .0006])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MNIST 2D Manifold Visualization\n{}".format(method))
    if save:
        fig.savefig(fname)
    if show:
        fig.show()



print("Loading manifold")
manifold = load_manifold("mnist-lle-60000-2d-2.npy")
print("Loading embedding")
embedding = load_embedding("mnist-lle-60000-2d-2.pkl")

adv_examples = []
for i in range(10):
    adv_x = np.load("datasets/test/{}.npy".format(str(i)))
    print("Transforming example")
    m_x = embedding.transform(adv_x.reshape(1, 28*28))
    adv_examples.append(m_x[0])

plot_manifold(manifold, 
                "Locally Linear Embedding\nAdversarial Examples", 
                show=True,
                save=True, 
                fname="lle-test-viz-0-10",
                adv_x = adv_examples)
plt.show()
