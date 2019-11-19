import tensorflow as tf
import numpy as np


def generate_adversarial(input_image, input_label, model):
    # from https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

    # Reshape example to add a single dimension at the beginning for tensor
    input_image = np.reshape(
        input_image,
        (1, input_image.shape[0], input_image.shape[1], input_image.shape[2]),
    )
    input_label = np.reshape(input_label, (1, input_label.shape[0]))

    # Create tensor from input
    input_tensor = tf.convert_to_tensor(input_image)

    # Get loss of error
    with tf.GradientTape() as tape:
        prediction = model(input_tensor)
        loss_val = tf.compat.v1.losses.softmax_cross_entropy(
            input_label, logits=prediction
        )

    # Get gradient with respect to the input tensor
    gradient = tf.gradients(loss_val, input_tensor)

    # Find direction of the gradient and return
    signed_grad = tf.sign(gradient)
    return signed_grad

