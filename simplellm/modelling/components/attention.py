import math
import torch
from torch import nn
import torch.nn.functional as F

###########################
# Vanilla Implementations #
###########################

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, nheads, ctx_size, dropout=0.0, causal=True):
        super().__init__()
        assert in_channels % nheads == 0

        self.W_Q = nn.Linear(in_channels, in_channels, bias=False)
        self.W_K = nn.Linear(in_channels, in_channels, bias=False)
        
        self.W_V = nn.Linear(in_channels, in_channels, bias=False)
        self.W_O = nn.Linear(in_channels, in_channels, bias=False)

        if causal:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones((ctx_size, ctx_size))).reshape(
                    1, 1, ctx_size, ctx_size
                )
            )
        self.causal = causal
        
        self.head_dim = in_channels // nheads
        self.nheads = nheads
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, in_channels = x.size()
        
        Q = self.W_Q(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (1.0 * math.sqrt(self.head_dim))

        if self.causal:
            attn = attn.masked_fill(self.mask[:,:,:seq_len,:seq_len]==0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ V
        out = out.transpose(1, 2).reshape(batch_size, seq_len, in_channels)
        out = self.W_O(out)

        return out


class MultiQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        latent_channels,
        in_channels,
        qk_channels=None,
        v_channels=None,
        out_channels=None,
        nheads=1,
        dropout=0.0
    ):
        super().__init__()

        # we want to default to latent channels for q/k/v
        if qk_channels is None:
            qk_channels = latent_channels
        if v_channels is None:
            v_channels = qk_channels
        if out_channels is None:
            # not sure why deepmind code defaults to v_channels since we want
            # the final channel number to match latent channels regardless
            out_channels = latent_channels

        assert qk_channels % nheads == 0
        assert v_channels % nheads == 0

        self.ln_1atent = nn.LayerNorm(latent_channels)
        self.ln_input = nn.LayerNorm(in_channels)

        self.W_Q = nn.Linear(latent_channels, qk_channels, bias=False)
        self.W_K = nn.Linear(in_channels, qk_channels, bias=False)
        
        self.W_V = nn.Linear(in_channels, v_channels, bias=False)
        self.W_O = nn.Linear(v_channels, out_channels, bias=False)

        self.v_channels = v_channels

        self.qk_head_dim = qk_channels // nheads 
        self.v_head_dim = v_channels // nheads
        
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent_q, input_kv):
        batch_size, input_seq_len, _ = input_kv.size()
        _, latent_seq_len, _ = latent_q.size()

        Q = self.W_Q(latent_q).reshape(batch_size, latent_seq_len, self.nheads, self.qk_head_dim).transpose(1, 2)
        K = self.W_K(input_kv).reshape(batch_size, input_seq_len, self.nheads, self.qk_head_dim).transpose(1, 2)
        V = self.W_V(input_kv).reshape(batch_size, input_seq_len, self.nheads, self.v_head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (1.0 * math.sqrt(self.qk_head_dim))
        attn = F.softmax(attn, dim=-1)
        
        attn = self.dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).reshape(batch_size, latent_seq_len, self.v_channels)
        out = self.W_O(out)

        return out


class ALiBiAttention(nn.Module):
    def __init__(self, in_channels, nheads, ctx_size, dropout=0.0):
        super().__init__()
        assert in_channels % nheads == 0

        self.W_Q = nn.Linear(in_channels, in_channels, bias=False)
        self.W_K = nn.Linear(in_channels, in_channels, bias=False)
        
        self.W_V = nn.Linear(in_channels, in_channels, bias=False)
        self.W_O = nn.Linear(in_channels, in_channels, bias=False)

        self.register_buffer("m", get_alibi_slope(nheads))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones((ctx_size, ctx_size))).reshape(
                1, 1, ctx_size, ctx_size
            )
        )
        
        self.head_dim = in_channels // nheads
        self.nheads = nheads
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, in_channels = x.size()
        
        Q = self.W_Q(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (1.0 * math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:,:,:seq_len,:seq_len]==0, float('-inf'))
        
        bias = (self.m * get_relative_positions(seq_len).to(x.device)).unsqueeze(0)
        attn = attn + bias
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ V
        out = out.transpose(1, 2).reshape(batch_size, seq_len, in_channels)
        out = self.W_O(out)

        return out


class RotaryAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

################################
# Fused-Kernel Implementations #
################################

class FlashAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ALiBiFlashAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


#########
# utils #
#########

def get_relative_positions(seq_len: int) -> torch.tensor:
    ''' adopted from https://github.com/jaketae/alibi/blob/main/alibi/attention.py '''
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    
    return (x - y).clamp_max_(0)


def get_alibi_slope(num_heads):
    ''' adopted from https://github.com/jaketae/alibi/blob/main/alibi/attention.py '''
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )