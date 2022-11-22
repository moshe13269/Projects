import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization


class PositionalEmbedding(tf.keras.layers.Layer):
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

        super(PositionalEmbedding, self).__init__()

        self.activation = tf.keras.layers.Activation(activation)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()

        if dim_conv == 1:  ##
            self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                                               activation=activation, padding='same')
        else:
            self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                               activation=activation, padding='same')

    def call(self, inputs, **kwargs):
        return self.layer_norm(self.add([self.conv(inputs), inputs]))


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class EncoderTransformer(tf.keras.layers.Layer):
    num_layers: int
    d_model: int
    num_attention_heads: int
    dff: int
    input_vocab_size: int
    dropout_rate: float
    dim_conv: int
    activation: str

    def __init__(self,
                 # *,
                 num_layers,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 activation: str,
                 dropout_rate=0.1,
                 dim_conv=1,
                 ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(kernel_size=3,
                                                 filters=d_model,
                                                 stride=1,
                                                 dim_conv=dim_conv,
                                                 activation=activation,
                                                 )

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_attention_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm_teacher = InstanceNormalization()

        self.add = tf.keras.layers.Add()
        self.subtract = tf.keras.layers.Subtract()

    def call(self, x, top_k_transformer=None):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        if top_k_transformer is None:
            for i in range(self.num_layers):
                x = self.enc_layers[i](x)

            return x  # Shape `(batch_size, seq_len, d_model)`.

        else:

            top_k_layers = []
            index_k0 = len(self.enc_layers) - top_k_transformer
            counter = 0

            for encoder_layer in self.enc_layers:

                x = self.layer_norm_teacher(encoder_layer(x))

                counter += 1

                if counter >= index_k0:
                    top_k_layers.append(x)
            
            return self.subtract([self.add(top_k_layers), top_k_transformer])


if __name__ == '__main__':
    encoder = EncoderTransformer(num_layers=12, d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1,
                                 activation='gelu')
    # encoder = EncoderLayer(d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1)
    # position = ConvPosEncoding(3, 512, 1, 1, 'gelu')
    data = tf.random.normal((2, 200, 512))
    # output1 = position(data, training=True)
    output = encoder(data, top_k_transformer=None)
    print(output.shape)
    print(tf.reduce_max(output))
