import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
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


def clac_conv_output(conv_layers_params, inputs_size, p):
    w = inputs_size

    for i, layers_param in enumerate(conv_layers_params):
        (dim, kernel, stride) = layers_param

        w = clac_conv_layer_output(w, kernel, stride, p)

    return w


def flat_list(list_of_list):
    flatten_list = []
    if type(list_of_list[0][0]) == int:
        return list_of_list
    for sublist in list_of_list:
        for item in sublist:
            flatten_list.append(item)
    return flatten_list


def repeated_conv_layers(conv_layers, num_duplicate_layer):
    repeated_list = [list(conv_layers[i]) * num_duplicate_layer[i] for i in range(len(num_duplicate_layer))]
    return flat_list(repeated_list)


def check_names(path):
    if os.path.exists(os.path.dirname(path)):
        if path.endswith('.npy'):
            file = np.load(path)
            return file.shape[0]
        elif path.endswith('.wav'):
            _, file = wavfile.read(path)
            return file.shape[0]
    return path


def outputs_conv_size(conv_layers, num_duplicate_layer, inputs_size, p, avg_pooling):
    """
    :param inputs_size: int - the input size or path to input size (which will load)
    """
    inputs_size = check_names(inputs_size)
    layers_params = repeated_conv_layers(conv_layers, num_duplicate_layer)
    if avg_pooling:
        return int(clac_conv_output(layers_params, inputs_size, p) / 2)
    return int(clac_conv_output(layers_params, inputs_size, p))


def init_weight_model(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        y = m.in_channels
        m.weight.data.normal_(0.0, 1.)# / np.sqrt(y))
        m.bias.data.fill_(0.)

    elif isinstance(m, nn.Linear):
        y = m.in_features
        m.weight.data.normal_(0.0, 1.)# /np.sqrt(y))
        m.bias.data.fill_(0.)


def load_model(path2load_model):
    pass