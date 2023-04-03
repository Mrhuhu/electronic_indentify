import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q.transpose(-1, -2), K) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V.transpose(-1, -2))  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, dropout):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model)  # 新增，为在cpu上训练，不用每次屏蔽.cuda()

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)


        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V)
        context = context.permute(0, 3, 1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.dropout(self.fc(context))  # [batch_size, len_q, d_model]
        outputs = nn.LayerNorm(self.d_model)(output + residual)
        return outputs


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(mlp_ratio * d_model, d_model, bias=False)
        )
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = self.dropout(output)
        return nn.LayerNorm(self.d_model)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super(EncoderLayer, self).__init__()
        self.d_v = d_model // n_heads
        self.d_k = d_model // n_heads
        self.enc_self_attn = MultiHeadAttention(d_model, self.d_k, n_heads, self.d_v, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_inputs, value_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, value_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = self.dropout(enc_outputs)
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, enc_inputs, key_inputs):
        enc_inputs = self.layer(enc_inputs, key_inputs)
        return enc_inputs


class Generator(nn.Module):
    def __init__(self, n_heads, mlp_ratio, resolution, num_classes, dropout):
        super(Generator, self).__init__()

        self.resolution = resolution
        self.conv_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=3, padding=1)

        self.encoder_layer = Encoder(self.resolution, n_heads, mlp_ratio, dropout)
        self.mlp_out = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, inputs):
        conv_input = self.conv_layer(inputs)
        enc_inputs = conv_input.view(conv_input.size(0), -1, self.resolution)
        batch_size, seq_len, dim = enc_inputs.shape
        transformer_buffer = []
        transformer_input_units = enc_inputs[:, 0, :]
        for layer in range(seq_len):
            if layer == 0:
                transformer_units_output = self.encoder_layer(transformer_input_units.unsqueeze(1),
                                                              transformer_input_units.unsqueeze(1))
                transformer_buffer.append(transformer_units_output)
            else:
                transformer_units_output = self.encoder_layer(enc_inputs[:, layer, :].unsqueeze(1),
                                                              transformer_units_output)
                transformer_buffer.append(transformer_units_output)

        transformer_output_unit = torch.hstack(transformer_buffer)
        transformer_output = transformer_output_unit.view(batch_size, -1)
        transformer_output = transformer_output + conv_input.squeeze(1)  # 残差连接
        output = self.mlp_out(transformer_output)
        return output


def count(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == '__main__':
    g = Generator(n_heads=4, mlp_ratio=4, resolution=250, num_classes=10, dropout=0.)
    cnt = count(g)
    z = torch.randn(4, 1, 3000)
    e = g(z)
