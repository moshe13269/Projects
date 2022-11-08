
import tensorflow as tf


def point_wise_feed_forward_network(
        d_model,  # Input/output dimensionality.
        dff  # Inner-layer dimensionality.
):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # Shape `(batch_size, seq_len, dff)`.
        tf.keras.layers.Dense(d_model),  # Shape `(batch_size, seq_len, d_model)`.
        tf.keras.layers.Activation('gelu') ####
    ])


class ConvPosEncoding(tf.keras.layers.Layer):
    filters: int
    kernel_size: int
    stride: int
    normalization: bool
    dim_conv: int
    activation: str

    def __init__(self,
                 kernel_size: int,
                 filters: int,
                 stride: int,
                 dim_conv: int,
                 activation: str,
                 ):

        super(ConvPosEncoding, self).__init__()

        # self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
        #                                    activation=activation, padding='same')
        if dim_conv == 1: ##
            self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                                               activation=activation, padding='same')
        else:
            self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                               activation=activation, padding='same')

    def call(self, inputs, **kwargs):
        return self.conv(inputs)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 dropout_rate=0.1
                 ):
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention.
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,  # Size of each attention head for query Q and key K.
            dropout=dropout_rate,
        )
        # Point-wise feed-forward network.
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer normalization.
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.activation = tf.keras.layers.Activation('gelu')

    def call(self, x, training):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=x,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=None,  # A boolean mask that prevents attention to certain positions.
            training=training,  # A boolean indicating whether the layer should behave in training mode.
        )

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layer_norm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layer_norm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return self.activation(out2)  ####


class TransformerEncoder(tf.keras.Model):
    num_layers: int
    d_model: int
    num_attention_heads: int
    dff: int
    input_vocab_size: int
    dropout_rate: float
    dim_conv: int

    def __init__(self,
                 # *,
                 num_layers,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 dropout_rate=0.1,
                 dim_conv=1
                 ):

        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.conv_pos_encoding = ConvPosEncoding(kernel_size=3, filters=self.d_model, stride=1, dim_conv=dim_conv,
                                                 activation='gelu')

        self.layer_norm = tf.keras.layers.LayerNormalization()

        # Encoder layers.
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                dff=dff,
                dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        # Dropout.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, top_k_transformer=None):

        x = self.layer_norm(x + self.conv_pos_encoding(x))
        x = self.dropout(x, training=training)

        # N encoder layers.
        if top_k_transformer is None:
            for encoder_layer in self.enc_layers:
                x = encoder_layer(x, training=training)
            return x
        else:
            top_k_layers = tf.zeros_like(x)
            index_k0 = len(self.enc_layers) - top_k_transformer
            counter = 0
            for encoder_layer in self.enc_layers:
                counter += 1
                x = encoder_layer(x, training=training)
                if counter >= index_k0:
                    top_k_layers += x
            return top_k_layers / top_k_transformer  #x / top_k_transformer  # Shape `(batch_size, input_seq_len, d_model)`.


if __name__ == '__main__':
    encoder = TransformerEncoder(num_layers=24, d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1)
    # encoder = EncoderLayer(d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1)
    position = ConvPosEncoding(3, 512, 1, 1, 'gelu')
    data = tf.random.normal((10, 200, 512))
    output1 = position(data, training=True)
    output2 = encoder(data, training=True, top_k_transformer=8)
    print(output1.shape, output2.shape)
    print(tf.reduce_mean(output1), tf.reduce_mean(output2))
