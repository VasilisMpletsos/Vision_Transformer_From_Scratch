import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dimension: int = 16, n_heads: int = 4):
        super().__init__()
        self.dimension = dimension
        self.n_heads = n_heads

        assert dimension % n_heads == 0, (
            f"Can't divide dimension {dimension} into {n_heads} heads"
        )

        head_dimension = dimension // n_heads
        self.head_dimension = head_dimension
        self.softmax = nn.Softmax(dim=-1)

        self.q_mappings = nn.ModuleList(
            [nn.Linear(head_dimension, head_dimension) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(head_dimension, head_dimension) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(head_dimension, head_dimension) for _ in range(self.n_heads)]
        )

    def forward(self, x):
        result = []
        for sequence in x:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                chunk_sequence = sequence[
                    :, head * self.head_dimension : (head + 1) * self.head_dimension
                ]
                q, k, v = (
                    q_mapping(chunk_sequence),
                    k_mapping(chunk_sequence),
                    v_mapping(chunk_sequence),
                )
                attention = self.softmax(q @ k.T) / (self.head_dimension**0.5)
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyEncoderBlock(nn.Module):
    def __init__(self, dimension: int = 16, n_heads: int = 4, mlp_ratio=4):
        super().__init__()
        self.dimension = dimension
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(dimension)
        self.mhsa = MultiHeadAttention(dimension, n_heads)

        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, mlp_ratio * dimension),
            nn.GELU(),
            nn.Linear(mlp_ratio * dimension, dimension),
        )

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
