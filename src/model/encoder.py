import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, LayerNorm, Linear, Module, Sequential


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


class ConvolutionalTokenEmbedding(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
    ) -> None:
        super().__init__()

        self.conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.layer_norm = LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        B, C, H, W = x.shape
        # Flatten the tensor at every channel
        x = x.reshape(B, C, -1)
        # Send the channel last
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        return x, H, W


class ConvolutionalAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: int | tuple,
        stride_q: int | tuple,
        stride_kv: int | tuple,
        padding_q: int | tuple,
        padding_kv: int | tuple,
        qk_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        cls_token: Tensor | None = None,
    ):
        super().__init__()
        self.cls_token = cls_token
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.conv_proj_q = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                stride=stride_q,
                padding=padding_q,
                groups=embed_dim,
            ),
            nn.BatchNorm2d(num_features=embed_dim),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
        )
        self.conv_proj_k = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                stride=stride_kv,
                padding=padding_kv,
                groups=embed_dim,
            ),
            nn.BatchNorm2d(num_features=embed_dim),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
        )
        self.conv_proj_v = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                stride=stride_kv,
                padding=padding_kv,
                groups=embed_dim,
            ),
            nn.BatchNorm2d(num_features=embed_dim),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
        )

        self.qk_dropout = nn.Dropout(p=qk_dropout)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x: Tensor):
        B, C, _, _ = x.shape
        q = self.conv_proj_q(x).reshape(B, C, -1).permute(0, 2, 1)
        k = self.conv_proj_k(x).reshape(B, C, -1).permute(0, 2, 1)
        v = self.conv_proj_v(x).reshape(B, C, -1).permute(0, 2, 1)

        if self.cls_token is not None:
            q = torch.cat([self.cls_token, q], dim=1)
            k = torch.cat([self.cls_token, k], dim=1)
            v = torch.cat([self.cls_token, v], dim=1)

        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.qk_dropout(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        attn = self.attn_dropout(self.proj(attn))
        return attn
