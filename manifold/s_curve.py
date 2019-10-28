import numpy as np
from sklearn import datasets, manifold
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.spatial import distance
import keras
import tensorflow as tf


def generate_dataset():
    np.random.seed(100)
    samples, locations = datasets.make_s_curve(1000)
    return samples, locations

def learn_manifold(data):
    isomap = manifold.Isomap()
    data_manifold = isomap.fit_transform(data[0])
    return data_manifold

def generate_model(samples, labels):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(30))
    model.add(keras.layers.Activation("sigmoid"))
    model.add(keras.layers.Dense(2))
    # model.add(keras.layers.Reshape((-1, 2)))
    model.add(keras.layers.Activation("softmax"))
    model.build((1, 1000, 3))
    model.summary()

    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.SGD(),
        metrics=["accuracy"]
    )

    return model

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

def adversarial_example(samples, labels, model):
    loss_object = keras.losses.CategoricalCrossentropy()
    input_sample = (np.array([samples[0]]))
    input_tensor = tf.convert_to_tensor(input_sample)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model.predict(input_sample)
        print(type(labels[0]), type(prediction))
        loss = loss_object([labels[0]], prediction)
    gradient = tape.gradient((loss), input_tensor)
    print(loss, input_tensor)
    print(gradient)
    signed_gradient = tf.sign(gradient)
    return signed_gradient



def main():
    samples, locations = generate_dataset()
    labels = []
    # make the labels be 0 and 1, based on positive or negative value on the z axis
    for s in samples:
        if s[2] < 0:
            labels.append(0)
        else:
            labels.append(1)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(samples[:,0], samples[:,1], samples[:,2], c=labels)

    samples = np.array([samples])
    labels = np.array([keras.utils.to_categorical(labels)])
    print(np.shape(samples))
    print(np.shape(labels))

    data_manifold = learn_manifold(samples)
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(data_manifold[:,0], data_manifold[:,1])
    # plt.show()

    model = generate_model(samples, labels)
    model.fit(samples, labels, epochs = 10)


    evil = adversarial_example(samples, labels, model)
    print(evil)


if __name__ == "__main__":
    main()