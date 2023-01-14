import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Conv1D, Dense, Layer


# num_heads = 8
# num_layers = 6
# d_model = 512
# d_ff = 2048
# d_k = d_model // num_heads
# drop_out_rate = 0.1
# num_epochs = 10
# beam_size = 8

class EncoderLayer(Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout, activation, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads, dropout=dropout
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.ffn = FFN(d_model, d_ff, dropout, activation)

    def call(self, inputs, **kwargs):
        x = self.ln1(inputs)
        x = x + self.dropout1(self.mha(x))
        x = self.ln2(x)
        x = x + self.dropout2(self.fc1(x))
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
    def __init__(self, dropout):
        super().__init__()
        self.ln1 = LayerNormalization()
        self.masked_mha = MultiHeadAttention()
        self.dropout1 = Dropout(dropout)

        self.ln2 = LayerNormalization()
        self.mha = MultiHeadAttention()
        self.dropout2 = Dropout(dropout)

        self.ln3 = LayerNormalization()
        self.feed_forward = FFN()
        self.dropout3 = Dropout(dropout)

    def forward(self, x, e_output, e_mask, d_mask):
        x_1 = self.ln1(x)  # (B, L, d_model)
        x = x + self.dropout1(
            self.masked_mha(x_1, x_1, x_1, attention_mask=d_mask)
        )  # (B, L, d_model)
        x_2 = self.ln2(x)  # (B, L, d_model)
        x = x + self.dropout2(
            self.mha(x_2, e_output, e_output, attention_mask=e_mask)
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
        self.ln = LayerNormalization(epsilon=1e-6)

    def forward(self, x):
        return self.layer_norm(self.conv(x) + x)


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

    def forward(self, x):
        x = x * tf.math.sqrt(self.d_model)  # (B, L, d_model)
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
        x = self.pos_encodeing(inputs)

        x = self.dropout(x)

        for encoder in self.encoder_blocks:
            x = encoder(x)
        return x


class TransformerDecoder(Layer):

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
        x = self.pos_encodeing(inputs)

        x = self.dropout(x)

        for encoder in self.encoder_blocks:
            x = encoder(x)
        return x