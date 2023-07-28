import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.cuda() == 0, -1e9)  # (maskc == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_output, embeddings_param, tgt_mask):
        attn_output = self.self_attn(tgt_output, tgt_output, tgt_output, tgt_mask)
        x = self.norm1(tgt_output + self.dropout(attn_output))

        attn_output = self.cross_attn(x, embeddings_param, embeddings_param)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Conv1DLayer(nn.Module):
    def __init__(self, input_shape, d_model=512, num_conv_layers=7, dropout=0.1, kernel=3, stride=1, padding=1):
        super(Conv1DLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        def make_conv_layer(in_channel):
            return nn.Conv1d(in_channels=in_channel,
                             out_channels=d_model,
                             kernel_size=kernel,
                             stride=stride,
                             bias=True,
                             padding=padding)

        self.conv_layers = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            if i != 0:
                self.conv_layers.append(
                    nn.Sequential(
                        make_conv_layer(d_model),
                        nn.Dropout(p=dropout),
                        nn.LayerNorm([d_model, input_shape[0]], elementwise_affine=True, eps=1e-6),
                        nn.ReLU(),
                    )
                )
            else:
                self.conv_layers.append(
                    nn.Sequential(
                        make_conv_layer(input_shape[1]),
                        nn.Dropout(p=dropout),
                        nn.LayerNorm([d_model, input_shape[0]], elementwise_affine=True, eps=1e-6),
                        nn.ReLU(),
                    )
                )

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        for layer in self.conv_layers:
            x = layer.forward(x)

        x = torch.permute(x, (0, 2, 1))

        return x


class TransformerD(nn.Module):
    """
    casual decoder transformer
    params:
    input shape: Tuple: (int, int) = (time_length, frequency)
    output shape: same as input  shape

    forward:
    src: param vector
    tgt: spectrogram
    """

    def __init__(self, d_model, num_heads, num_layers, d_ff, input_shape: tuple[int, int],
                 dropout=0.1, path2csv=None, num_quant_params=None):
        super(TransformerD, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, input_shape[0])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc_embeddings = nn.Linear(1, d_model)
        self.fc_output = nn.Linear(d_model, input_shape[1])
        self.dropout = nn.Dropout(dropout)

        self.conv1d = Conv1DLayer(input_shape=input_shape)

        self.path2csv = path2csv
        self.num_quant_params = num_quant_params
        self.embedding = None
        self.init_embedding_layer(d_model)

    def init_embedding_layer(self, d_model):
        if self.path2csv is not None:
            import pandas as pd
            csv = pd.read_csv(self.path2csv)
            params_quant_arr = []
            for key in csv.keys():
                len_row_set = len(set(csv[key]))
                if len_row_set < 500:
                    params_quant_arr.append(len_row_set)
            self.num_quant_params = params_quant_arr

        self.embedding = nn.ModuleList(
            [nn.Embedding(num_embeddings=self.num_quant_params[i], embedding_dim=d_model, max_norm=True)
             for i in range(len(self.num_quant_params))]
        )

    def generate_mask(self, tgt):
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = torch.ones(tgt.shape[0], 1, seq_length, 1).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask

    def forward(self, decoder_inputs: torch.Tensor, condition_vector: torch.Tensor):
        """
        src: param vector with shape (batch, N) where N is the number of param in the synth (row from the cvs)
        tgt: spectrogram of the target outputs wav: (batch, time_axis, freq_axis)
        """
        embedding_decoder_inputs = []
        for i in range(len(self.embedding)):
            embedding_decoder_inputs.append(
                self.embedding[i](condition_vector[:, i:i+1])
            )
        embedding_decoder_inputs = torch.cat(embedding_decoder_inputs, dim=1)
        masking = self.generate_mask(decoder_inputs)

        decoder_hidden = self.dropout(self.positional_encoding(self.conv1d(decoder_inputs)))

        for dec_layer in self.decoder_layers:
            decoder_hidden = dec_layer(decoder_hidden, embedding_decoder_inputs, masking)

        decoder_output = torch.nn.functional.relu(self.fc_output(decoder_hidden))

        return decoder_output


if __name__ == "__main__":

    trans_decoder = TransformerD(d_model=512,
                                 num_heads=16,
                                 num_layers=12,
                                 d_ff=2048,
                                 input_shape=(130, 129),
                                 dropout=0.1,
                                 path2csv=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\Data_custom_synth.csv')
    decoder_inputs = torch.normal(mean=torch.zeros(64, 130, 129))
    condition_vector = torch.randint(0, 3, size=(64, 9, 31)).type(torch.float32)
    output_dec = trans_decoder(decoder_inputs=decoder_inputs, condition_vector=condition_vector)
    print(output_dec.shape)