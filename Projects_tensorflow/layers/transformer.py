import tensorflow as tf
from tensorflow.python.keras.layers import MultiHeadAttention, Dropout, \
    Conv1D, Dense, Layer


class EncoderLayer(Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout, activation, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.mha = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads, dropout=dropout
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.ffn = FFN(d_model, d_ff, dropout, activation)

    def call(self, inputs, **kwargs):
        x = inputs #, e_mask = inputs
        x = self.ln1(x)
        x = x + self.dropout1(self.mha(query=x, value=x, key=x)) #, attention_mask=e_mask))
        x = self.ln2(x)
        x = x + self.dropout2(self.ffn(x))
        return x


class FFN(Layer):

    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()
        self.fc1 = Dense(d_ff, activation=activation)
        self.fc2 = Dense(d_model, activation=None)
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = self.fc1(inputs)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DecoderLayer(Layer):
    def __init__(self, d_model, d_ff, dropout, activation, num_heads):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.masked_mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout1 = Dropout(dropout)

        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout2 = Dropout(dropout)

        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = FFN(d_model, d_ff, dropout, activation)
        self.dropout3 = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x, e_output, d_mask = inputs #x, e_output, e_mask, d_mask = inputs

        x_1 = self.ln1(x)  # (B, L, d_model)

        x = x + self.dropout1(
            self.masked_mha(x_1, x_1, x_1, attention_mask=d_mask) # use_causal_mask=True) #
        )  # (B, L, d_model)

        x_2 = self.ln2(x)  # (B, L, d_model)

        x = x + self.dropout2(
            self.mha(x_2, e_output, e_output, attention_mask=None) #e_mask)
        )  # (B, L, d_model)

        x_3 = self.ln3(x)  # (B, L, d_model)

        x = x + self.dropout3(self.feed_forward(x_3))  # (B, L, d_model)

        return x  # (B, L, d_model)


class LearnablePositionalEncoder(Layer):
    def __init__(self, d_model, activation='relu', kernel_size=3, stride=1):
        super().__init__()
        self.d_model = d_model
        self.conv = Conv1D(filters=d_model, kernel_size=kernel_size, strides=stride,
                           activation=activation, padding='same')
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        return self.ln(outputs + inputs)


class PositionalEncoder(Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.d_model = d_model

        # Make initial positional encoding matrix with 0
        pe_matrix = tf.zeros(seq_len, d_model)  # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = tf.math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = tf.math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
        self.positional_encoding = tf.stop_gradient(pe_matrix)  # .requires_grad_(False)

    def call(self, inputs, **kwargs):
        x = inputs * tf.math.sqrt(self.d_model)  # (B, L, d_model)
        x = x + self.positional_encoding  # (B, L, d_model)

        return x


class TransformerEncoder(Layer):

    def __init__(self,
                 num_transformer_blocks,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout,
                 activation,
                 **kwargs):
        super().__init__(**kwargs)

        self.encoder_blocks = [EncoderLayer(d_model, num_heads, d_ff, dropout, activation) for
                               _ in range(num_transformer_blocks)]

        self.pos_encodeing = LearnablePositionalEncoder(d_model,
                                                        activation='gelu')
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs #x, e_mask = inputs
        x = self.pos_encodeing(x)

        x = self.dropout(x)

        for encoder in self.encoder_blocks:
            x = encoder(x) #x = encoder([x, e_mask])
        return x


class TransformerDecoder(Layer):

    def __init__(self,
                 num_transformer_decoder_blocks,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout,
                 activation,
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_blocks = [DecoderLayer(d_model, d_ff, dropout, activation, num_heads) for
                               _ in range(num_transformer_decoder_blocks)]

        self.pos_encodeing = LearnablePositionalEncoder(d_model,
                                                        activation='gelu')
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x, e_output, d_mask = inputs #x, e_output, e_mask, d_mask = inputs
        x = self.pos_encodeing(x)

        x = self.dropout(x)

        for decoder in self.decoder_blocks:
            x = decoder([x, e_output, d_mask])
        return x


class Transformer(Layer):
    def __init__(self,
                 num_transformer_blocks,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout,
                 activation,
                 decoder=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.transformer_encoder = TransformerEncoder(num_transformer_blocks,
                                                      d_model,
                                                      num_heads,
                                                      d_ff,
                                                      dropout,
                                                      activation)

        self.decoder = decoder
        if self.decoder:
            self.transformer_decoder = TransformerDecoder(
                num_transformer_blocks,
                d_model,
                num_heads,
                d_ff,
                dropout,
                activation)

        self.fc = Dense(d_model, activation=activation)

    def call(self, inputs, **kwargs):
        x, d_mask = inputs

        enc_outputs = self.transformer_encoder(x)

        if self.decoder:
            x = self.transformer_decoder([x, enc_outputs, d_mask])

            x = self.fc(x)

            return x, enc_outputs
        return enc_outputs


if __name__ == '__main__':
    transformer = Transformer(
        num_transformer_blocks=2,
        d_model=512,
        num_heads=12,
        d_ff=3072,
        dropout=0.1,
        activation='relu'
    )

    data = tf.random.normal((2, 50, 512))
    mask_e = tf.where(tf.random.uniform((2, 50, 1)) >= 0.7, 1., 0.)
    mask_d = tf.where(tf.random.uniform((2, 50, 1)) >= 0.7, 1., 0.)
    outputs = transformer([data, mask_e, mask_d])
    print(tf.reduce_mean(outputs[0]), tf.reduce_mean(outputs[1]))
    a = 0
