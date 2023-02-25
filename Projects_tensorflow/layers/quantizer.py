
import tensorflow as tf


class Layer(tf.keras.layers.Layer):
    def __init__(self, *,G, V, activation, tau, word_depth=512):
        super(Layer, self).__init__()

        self.word_depth = word_depth
        self.code_book = None
        self.G = G
        self.V = V
        self.fc1 = tf.keras.layers.Dense(units=self.G*self.V, activation=activation)
        self.fc2 = tf.keras.layers.Dense(units=self.word_depth, activation=activation)
        self.tau = tau

    def build(self, input_shape):
        self.code_book = self.add_weight("code_book",
                                         shape=[self.G, self.V, self.word_depth],
                                         trainable=True,
                                         initializer=tf.keras.initializers.Orthogonal)

    def call(self, data, **kwargs):
        logits = tf.transpose(tf.reshape(self.fc1(data), (data.shape[0], data.shape[1], self.V, self.G)), perm=[0,1,3,2])

        noise = -tf.math.log(-tf.math.log(tf.random.uniform(shape=logits.shape, maxval=1)))

        numertor_p_g_v = tf.math.exp(noise + logits)/self.tau

        p_g_v = numertor_p_g_v / tf.reduce_sum(numertor_p_g_v, axis=-1, keepdims=True)/self.tau

        y_hard = tf.cast(tf.math.greater_equal(p_g_v, tf.math.reduce_max(p_g_v, axis=-1, keepdims=True) - 10 ** -12),
                         p_g_v.dtype)

        p_gv_one_hot = tf.stop_gradient(y_hard - p_g_v) + p_g_v

        q_t = tf.math.multiply(tf.expand_dims(p_gv_one_hot, axis=-1),
                               tf.expand_dims(tf.expand_dims(self.code_book, 0), 0))

        q_t = tf.math.reduce_sum(q_t, axis=3)

        q_ts = tf.split(q_t, axis=2, num_or_size_splits=self.G)
        sub_words = tf.concat(q_ts, axis=-1)

        q = self.fc2(tf.squeeze(sub_words, axis=[2]))

        return p_g_v, q


if __name__ == '__main__':
    inputs = tf.random.normal(shape=(3, 20, 512))
    layers = Layer(G=2, V=13, activation=None, tau=2.)
    outputs = layers(inputs)
    print(outputs) #.shape)
