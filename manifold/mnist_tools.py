import keras


'''
Helper methods and variables for mnist models and manifolds
'''

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


# # Returns 4D np array (1, HEIGHT, WIDTH, 1)
# def tensor_to_numpy(t):
#     sess = K.get_session()
#     t_np = sess.run(t)

#     # Get rid of the extra dimension
#     t_np = t_np.reshape(1, HEIGHT, WIDTH, 1)
#     return t_np

def convert_to_model(seq_model):
    # From https://github.com/keras-team/keras/issues/10386
    input_layer = keras.layers.Input(batch_shape=seq_model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seq_model.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)
    funcmodel = keras.models.Model([input_layer], [prev_layer])

    return funcmodel