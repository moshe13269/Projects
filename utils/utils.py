
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


def clac_conv2d_output(conv_layers_params, inputs_size):
    w = inputs_size

    for i, layers_param in enumerate(conv_layers_params):
        (dim0, kernel0, stride0) = layers_param[0]
        (dim1, kernel1, stride1) = layers_param[1]

        w = clac_conv_layer_output(w, kernel0, stride0)
        w = clac_conv_layer_output(w, kernel1, stride1)

    return w


def clac_conv1d_output(conv_layers_params, inputs_size):
    w = inputs_size

    for i, layers_param in enumerate(conv_layers_params):
        (dim, kernel, stride) = layers_param

        w = clac_conv_layer_output(w, kernel, stride)

    return w
