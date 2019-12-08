from matplotlib import pyplot as plt
import numpy as np

### Not using yet, but trying to move plotting stuff out for clarity

def load_manifold(filename):
    manifold = np.load(filename)
    return manifold

# TODO: add colors?
def plot_manifold(manifold, method, show=True, save=False, fname=""):
    fig, ax = plt.subplots()
    ax.scatter(manifold[:, 0], manifold[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Manifold Visualization with {}".format(method))
    if save:
        fig.savefig(fname)
    if show:
        fig.show()
