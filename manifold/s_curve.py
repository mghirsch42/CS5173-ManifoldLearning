import numpy as np
from sklearn import datasets, manifold
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.spatial import distance


def generate_dataset():
    np.random.seed(100)
    samples, locations = datasets.make_s_curve(1000)
    return samples, locations


def detect(samples, new_example):
    closest_point = []
    closest_distance = 10
    for point in samples:
        if len(closest_point) == 0: 
            closest_point = point
            closest_distance = distance.euclidean(point, new_example)
        else:
            new_dist = distance.euclidean(point, new_example)
            if new_dist < closest_distance:
                closest_point = point
                closest_distance = distance.euclidean(point, new_example)
    if closest_distance > .5:
        return True, closest_point
    else:
        return False, closest_point


def main():
    samples, locations = generate_dataset()
    new_example = [3, 3, 3]
    evil, closest_point  = detect(samples, new_example)
    print(evil, closest_point)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(samples[:,0], samples[:,1], samples[:,2], c=locations)    
    ax.scatter3D(new_example[0], new_example[1], new_example[2], c="red", s=20)
    ax.plot3D([3, closest_point[0]], [3, closest_point[1]], [3, closest_point[2]])
    plt.show()


if __name__ == "__main__":
    main()