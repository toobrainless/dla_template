import torch
from torch import nn


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, dropout, embed_dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        # print("MultiHeadedSelfAttentionModule")
        # print(f"{x.shape=}")
        output = self.norm(x)
        # print(f"{output.shape=}")
        output = self.attention(output, output, output, need_weights=False)[0]
        # print(f"{output.shape=}")
        output = self.dropout(output) + x
        # print(f"{output.shape=}")
        return output
