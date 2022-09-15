# from abc import ABC
import tensorflow as tf
# from tensorflow import keras
# from typing import List, Tuple
# import tensorflow_addons as tfa
# from dataclasses import dataclass, field
# import keras_nlp
# from tensorflow.python.keras import Model
# from tensorflow.python.keras.layers import Layer as BaseLayer, Conv1D, Dropout


def point_wise_feed_forward_network(
        d_model,  # Input/output dimensionality.
        dff  # Inner-layer dimensionality.
):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # Shape `(batch_size, seq_len, dff)`.
        tf.keras.layers.Dense(d_model)  # Shape `(batch_size, seq_len, d_model)`.
    ])


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
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

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
        out1 = self.layernorm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2


class TransformerEncoder(tf.keras.layers.Layer):
    num_layers: int
    d_model: int
    num_attention_heads: int
    dff: int
    input_vocab_size: int
    dropout_rate: float

    def __init__(self,
                 *,
                 num_layers,
                 d_model,  # Input/output dimensionality.
                 num_attention_heads,
                 dff,  # Inner-layer dimensionality.
                 input_vocab_size,  # Input (Portuguese) vocabulary size.
                 dropout_rate=0.1
                 ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Embeddings.
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
        # Positional encoding.
        self.pos_encoding = positional_encoding(MAX_TOKENS, self.d_model)

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

    def call(self, x, training):
        seq_len = tf.shape(x)[1]

        # Sum up embeddings and positional encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        # Add dropout.
        x = self.dropout(x, training=training)

        # N encoder layers.
        for encoder_layer in self.enc_layers:
            x = encoder_layer(x, training=training)

        return x  # Shape `(batch_size, input_seq_len, d_model)`.


if __name__ == '__main__':
    encoder = EncoderLayer(d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1)
    data = tf.random.normal((10, 100, 512))
    output = encoder(data)
    print(output.shape)