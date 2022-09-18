
import tensorflow as tf
import tensorflow_addons as tfa


class FFNLayer(tf.keras.layers.Layer):

    dff: int
    activation: str

    def __init__(self, dff: int, activation: str):
        super(FFNLayer).__init__()
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation=activation)])
        self.instance_norm = tfa.layers.InstanceNormalization()

    def call(self, inputs, *args, **kwargs):
        return self.instance_norm(self.ffn(inputs))


class FFN(tf.keras.layers.Layer):

    dff: int
    activation: str
    num_layers: int

    def __init__(self, dff: int, activation: str, num_layers: int,):
        super(FFN).__init__()
        self.fnn_layers = [FFNLayer(dff, activation) for _ in range(num_layers)]
        self.num_layers = num_layers

    def call(self, inputs, *args, **kwargs):
        return [self.fnn_layers[index_layer](inputs[index_layer])
                for index_layer in range(self.num_layers)]

