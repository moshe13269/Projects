
import os
import numpy as np
from scipy.io import wavfile



def padding_same(k, s, w):
    """
    k = filter size
    S = stride
    W = input size
    results: P = ((S - 1) * W - S + K) / 2
    """
    return ((s - 1) * w - s + k) / 2


def clac_conv_layer_output(w, k, s, p=None):
    """
    W is the input volume
    K is the Kernel size
    P is the padding
    S is the stride
    """
    if p is None:
        p = padding_same(k, s, w)
    return ((w - k + 2 * p) / s) + 1  # [(w−k+2*p)/s] + 1


def clac_conv_output(conv_layers_params, inputs_size):
    w = inputs_size

    for i, layers_param in enumerate(conv_layers_params):
        (dim, kernel, stride) = layers_param

        w = clac_conv_layer_output(w, kernel, stride)

    return w


def flat_list(list_of_list):
    flatten_list = []
    for sublist in list_of_list:
        for item in sublist:
            flatten_list.append(item)
    return flatten_list


def repeated_conv_layers(conv_layers, num_duplicate_layer):
    repeated_list = [list(conv_layers[i]) * num_duplicate_layer[i] for i in range(len(num_duplicate_layer))]
    return flat_list(repeated_list)


def check_names(path):
    if not os.path.exists(os.path.dirname(path)):
        if path.endswith('.npy'):
            file = np.load(path)
            return file.shape[0]
        elif path.endswith('.wav'):
            _, file = wavfile.read(path)
            return file.shape[0]
    return path


def outputs_conv_size(conv_layers, num_duplicate_layer, inputs_size):
    """
    :param inputs_size: int - the input size or path to input size (which will load)
    """
    inputs_size = check_names(inputs_size)
    layers_params = repeated_conv_layers(conv_layers, num_duplicate_layer)
    return int(clac_conv_output(layers_params, inputs_size)/2)
