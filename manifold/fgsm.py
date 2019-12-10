import tensorflow as tf
import numpy as np
from keras import backend as K

'''
Methods for generating adversarial examples
'''

# Generates perturbations for an input image, label, and network model 
def generate_adversarial(input_image, input_label, epsilon, model):
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

    # Find direction of the gradient
    signed_grad = tf.sign(gradient)
    perturbations = tensor_to_numpy(signed_grad)

    # Create adversarial example
    adv_x = input_image + epsilon * perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    adv_x = tensor_to_numpy(adv_x)

    return adv_x

def generate_all(x_train, y_train, epsilon, model):
    adv_examples = []
    for i in range(len(x_train)):
        base_example = x_train[i]  # Dimensions (HEIGHT, WIDTH, 1)
        base_label = y_train[i]

        # Create perturbation values
        perturbations = generate_adversarial(base_example, base_label, model)
        np_pert = tensor_to_numpy(perturbations)

        # Create adversarial example
        adv_x = base_example + epsilon * perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        adv_x = tensor_to_numpy(adv_x)
        adv_examples.append(adv_x)
    
        np.save("adversarial examples/" + str(i), adv_x)

    return adv_examples


# Returns 4D np array (1, HEIGHT, WIDTH, 1)
def tensor_to_numpy(t):
    sess = K.get_session()
    t_np = sess.run(t)

    # Get rid of the extra dimension
    t_np = t_np.reshape(1, 28, 28, 1)
    return t_np
