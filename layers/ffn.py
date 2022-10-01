
import tensorflow as tf
import tensorflow_addons as tfa


class FFNLayer(tf.keras.layers.Layer):

    dff: int
    activation: str

    def __init__(self, dff: int, activation: str):
        super(FFNLayer, self).__init__()
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation=activation)])
        self.instance_norm = tfa.layers.InstanceNormalization()

    def call(self, inputs, *args, **kwargs):
        return self.instance_norm(self.ffn(inputs))


class FFN(tf.keras.layers.Layer):

    dff: int
    activation: str
    num_layers: int

    def __init__(self, dff: int, activation: str, num_layers: int,):
        super(FFN, self).__init__()
        self.fnn_layers = [FFNLayer(dff, activation) for _ in range(num_layers)]
        self.num_layers = num_layers

    def call(self, inputs, *args, **kwargs):
        return tf.concat([tf.expand_dims(self.fnn_layers[index_layer](inputs[index_layer]), axis=1)
                          for index_layer in range(self.num_layers)], axis=1)


if __name__ == '__main__':
   ffn = FFN(dff=512, activation='gelu', num_layers=1)
   input = tf.random.normal((6, 15, 512))
   output = ffn(input)
   print(output.shape)