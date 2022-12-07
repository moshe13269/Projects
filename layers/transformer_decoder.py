import tensorflow as tf
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

        if dim_conv == 1:  ##
            self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                                               activation=activation, padding='same')
        else:
            self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                               activation=activation, padding='same')

    def call(self, inputs, **kwargs):
        return self.layer_norm(self.conv(inputs) + inputs)


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


class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)#,
        # use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  

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


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class DecoderTransformer(tf.keras.layers.Layer):

    num_layers: int
    d_model: int
    num_attention_heads: int
    dff: int
    dropout_rate: float
    dim_conv: int
    activation: str

    def __init__(self, *,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 dff,
                 activation: str,
                 dim_conv=1,
                 dropout_rate=0.1):

        super(DecoderTransformer, self).__init__()
        self._name = 'DecoderTransformer'
        self.d_model = d_model
        self.num_layers = num_layers

        # self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
        #                                          d_model=d_model)

        self.pos_embedding = PositionalEmbedding(kernel_size=3,
                                                 filters=d_model,
                                                 stride=1,
                                                 dim_conv=dim_conv,
                                                 activation=activation,
                                                 )

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_attention_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

        self.layer_norm_teacher = InstanceNormalization()

        self.add = tf.keras.layers.Add()
        self.subtract = tf.keras.layers.Subtract()

    def call(self, x, context, top_k_transformer=None):

        # `x` is token-IDs shape (batch, target_seq_len)

        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        if top_k_transformer is None:

            for i in range(self.num_layers):
                x = self.dec_layers[i](x, context)

            self.last_attn_scores = self.dec_layers[-1].last_attn_scores

            # The shape of x is (batch_size, target_seq_len, d_model).
            return x

        else:

            top_k_layers = []
            index_k0 = len(self.dec_layers) - top_k_transformer
            counter = 0

            for decoder_layer in self.dec_layers:

                x = self.layer_norm_teacher(decoder_layer(x, context))

                counter += 1

                if counter >= index_k0:
                    top_k_layers.append(x)

            return tf.divide(self.add(top_k_layers), top_k_transformer)


