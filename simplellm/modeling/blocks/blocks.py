import math
import torch
from torch import nn
import torch.nn.functional as F

from components.attention import MultiHeadCrossAttention, MultiHeadAttention
from components.mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, nheads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_channels)
        self.attn = MultiHeadAttention(in_channels, nheads, dropout)
        self.ln_2 = nn.LayerNorm(in_channels)
        self.mlp = MLP(in_channels, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_channels, in_channels, nxheads, dropout=0.0):
        super().__init__()
        self.xattn = MultiHeadCrossAttention(
                    latent_channels,
                    in_channels,
                    nheads=nxheads,
                    dropout=dropout
        )
        self.mlp = MLP(latent_channels, dropout=dropout)
        
        self.ln_latent = nn.LayerNorm(latent_channels)
        self.ln_input = nn.LayerNorm(in_channels)
        self.ln_mlp = nn.LayerNorm(latent_channels)

    def forward(self, latents, x):
        latents = latents + self.xattn(self.ln_latent(latents), self.ln_input(x))
        latents = latents + self.mlp(self.ln_mlp(latents))

        return latents


class PerceiverBlock(nn.Module):
    def __init__(
        self,
        latent_channels,
        in_channels,
        nheads=8,
        nxheads=1,
        nlayers=4,
        dropout=0.0
    ):
        ''' PerceiverBlock is one CrossAttentionBlock followed by nlayer standard TransformerBlocks '''
        super().__init__()
        self.xattn_block = CrossAttentionBlock(latent_channels, in_channels, nxheads=nxheads, dropout=dropout)
        self.attn_blocks = nn.ModuleList([
            TransformerBlock(latent_channels, nheads=nheads, dropout=dropout)
            for _ in range(nlayers)
        ])

    def forward(self, latents, x):
        latents = self.xattn_block(latents, x)
        for block in self.attn_blocks:
            latents = block(latents)

        return latents