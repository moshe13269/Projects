import torch
from torch import nn
from torch.nn import MultiheadAttention, Dropout, LayerNorm, Linear, Conv1d, GELU, ReLU


class EncoderLayer(nn.Module):
    # shape: List[int, int, int]  # [C, H, W]

    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout,
                 activation
                 ):
        super().__init__()
        self.ln1 = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)
        self.ln2 = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)
        self.mha = nn.MultiheadAttention(batch_first=True,
                                         dropout=dropout,
                                         num_heads=num_heads,
                                         embed_dim=d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.ffn = FFN(d_model, d_ff, dropout, activation)

        if activation == 'gelu':
            self.activation = GELU()
        else:
            self.activation = ReLU()

    def forward(self, inputs):
        x = inputs  # , e_mask = inputs
        x = self.ln1(x)
        x = x + self.dropout1(self.mha(query=x, value=x, key=x))  # , attention_mask=e_mask))
        x = self.ln2(x)
        x = x + self.dropout2(self.activation(self.ffn(x)))
        return x


class FFN(nn.Module):

    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()
        self.fc1 = Linear(in_features=d_model, out_features=d_ff)
        self.fc2 = Linear(in_features=d_ff, out_features=d_model)

        self.dropout = Dropout(dropout)

        if activation == 'gelu':
            self.activation = GELU()
        else:
            self.activation = ReLU()

    def forward(self, inputs):
        x = self.activation(self.fc1(inputs))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 d_ff,
                 dropout,
                 activation,
                 num_heads):
        super().__init__()
        self.ln1 = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)
        self.masked_mha = MultiheadAttention(batch_first=True,
                                             dropout=dropout,
                                             num_heads=num_heads,
                                             embed_dim=d_model)
        self.dropout1 = Dropout(dropout)

        self.ln2 = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)
        self.mha = MultiheadAttention(batch_first=True,
                                      dropout=dropout,
                                      num_heads=num_heads,
                                      embed_dim=d_model)
        self.dropout2 = Dropout(dropout)

        self.ln3 = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)
        self.feed_forward = FFN(d_model, d_ff, dropout, activation)
        self.dropout3 = Dropout(dropout)

    def forward(self, inputs):
        x, e_output, d_mask = inputs  # x, e_output, e_mask, d_mask = inputs

        x_1 = self.ln1(x)  # (B, L, d_model)

        x = x + self.dropout1(
            self.masked_mha(x_1, x_1, x_1, mask=d_mask)  # use_causal_mask=True) #
        )  # (B, L, d_model)

        x_2 = self.ln2(x)  # (B, L, d_model)

        x = x + self.dropout2(
            self.mha(x_2, e_output, e_output)  # e_mask)
        )  # (B, L, d_model)

        x_3 = self.ln3(x)  # (B, L, d_model)

        x = x + self.dropout3(self.feed_forward(x_3))  # (B, L, d_model)

        return x  # (B, L, d_model)


class LearnablePositionalEncoder(nn.Module):

    def __init__(self,
                 d_model, activation='relu', kernel_size=3, stride=1):
        super().__init__()
        self.d_model = d_model
        self.conv = Conv1d(in_channels=d_model,
                           out_channels=d_model,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=1)

        if activation == 'gelu':
            self.activation = GELU()
        else:
            self.activation = ReLU()

        self.ln = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)

    def forward(self, inputs):
        output = self.activation(self.conv(inputs))
        return self.ln(output + inputs)


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.d_model = d_model

        # Make initial positional encoding matrix with 0
        pe_matrix = torch.zeros(seq_len, d_model)  # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = torch.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = torch.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
        with torch.no_grad:
            self.positional_encoding = pe_matrix  # .requires_grad_(False)

    def forward(self, inputs):
        x = inputs * torch.sqrt(self.d_model)  # (B, L, d_model)
        x = x + self.positional_encoding  # (B, L, d_model)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self,
                 num_transformer_blocks,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout,
                 activation):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(EncoderLayer(d_model, num_heads, d_ff, dropout, activation) for
                                            _ in range(num_transformer_blocks))

        self.pos_encoding = LearnablePositionalEncoder(d_model,
                                                       activation='gelu')
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)

    def forward(self, inputs):
        x = inputs  # x, e_mask = inputs
        x = self.pos_encoding(x)

        x = self.dropout(x)

        for encoder in self.encoder_blocks:
            x = encoder(x)  # x = encoder([x, e_mask])
        return self.layer_norm(x)


class TransformerDecoder(nn.Module):

    def __init__(self,
                 num_transformer_decoder_blocks,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout,
                 activation):
        super().__init__()

        self.decoder_blocks = nn.ModuleList(DecoderLayer(d_model, d_ff, dropout, activation, num_heads) for
                                            _ in range(num_transformer_decoder_blocks))

        self.pos_encoding = LearnablePositionalEncoder(d_model,
                                                       activation='gelu')
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm([d_model], elementwise_affine=True, eps=1e-6)

    def forward(self, inputs):
        x, e_output, d_mask = inputs  # x, e_output, e_mask, d_mask = inputs

        x = self.pos_encoding(x)
        x = self.dropout(x)

        for decoder in self.decoder_blocks:
            x = decoder([x, e_output, d_mask])
        return self.layer_norm(x)


class Transformer(nn.Module):
    def __init__(self,
                 num_transformer_blocks,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout,
                 activation):
        super().__init__()

        self.transformer_encoder = TransformerEncoder(num_transformer_blocks,
                                                      d_model,
                                                      num_heads,
                                                      d_ff,
                                                      dropout,
                                                      activation)

        self.transformer_decoder = TransformerDecoder(
            num_transformer_blocks,
            d_model,
            num_heads,
            d_ff,
            dropout,
            activation)

        if activation == 'gelu':
            self.activation = GELU()
        else:
            self.activation = ReLU()

        self.fc = Linear(in_features=d_model, out_features=d_model)

    def forward(self, inputs):
        x, d_mask = inputs

        enc_outputs = self.transformer_encoder(x)

        x = self.transformer_decoder([x, enc_outputs, d_mask])

        x = self.activation(self.fc(x))

        return x, enc_outputs


if __name__ == '__main__':
    transformer = Transformer(
        num_transformer_blocks=2,
        d_model=512,
        num_heads=12,
        d_ff=3072,
        dropout=0.1,
        activation='relu'
    )

    data = torch.normal(mean=0., std=1., size=(2, 50, 512))
    mask_e = torch.where(torch.rand((2, 50, 1)) >= 0.7, 1., 0.)
    mask_d = torch.where(torch.rand((2, 50, 1)) >= 0.7, 1., 0.)
    outputs = transformer([data, mask_e, mask_d])
    print(torch.mean(outputs[0]), torch.mean(outputs[1]))
    a = 0
