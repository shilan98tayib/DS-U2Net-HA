######################    HYBRID ATTENTION MECHANISM ######################
import torch
import torch.nn as nn


class TemporalSpectralCrossAttentionSegment(nn.Module):
    """
    Temporal-Spectral Attention Segment.

    Input shape:
        x: (B, C, T, F)

    Output shape:
        out: (B, C, T, F)
    """

    def __init__(self, in_channels: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        if in_channels % num_heads != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})."
            )

        # Temporal branch: attention over time for each frequency bin
        self.tmha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.t_ln1 = nn.LayerNorm(in_channels)
        self.t_ln2 = nn.LayerNorm(in_channels)

        self.t_cffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        )

        # Spectral branch: attention over frequency for each time frame
        self.smha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.s_ln1 = nn.LayerNorm(in_channels)
        self.s_ln2 = nn.LayerNorm(in_channels)

        self.s_cffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, T, F = x.shape

        # Temporal Attention Transformer (TAT)
        xt = x.permute(0, 3, 2, 1).reshape(B * F, T, C)
        xt_norm = self.t_ln1(xt)
        tmha_out, _ = self.tmha(xt_norm, xt_norm, xt_norm)
        xt = xt + tmha_out

        xt_norm = self.t_ln2(xt)
        xt4 = xt_norm.reshape(B, F, T, C).permute(0, 3, 2, 1)
        t_ffn = self.t_cffn(xt4)
        x_tat = xt4 + t_ffn

        # Spectral Attention Transformer (SAT)
        xs = x_tat.permute(0, 2, 3, 1).reshape(B * T, F, C)
        xs_norm = self.s_ln1(xs)
        smha_out, _ = self.smha(xs_norm, xs_norm, xs_norm)
        xs = xs + smha_out

        xs_norm = self.s_ln2(xs)
        xs4 = xs_norm.reshape(B, T, F, C).permute(0, 3, 1, 2)
        s_ffn = self.s_cffn(xs4)
        out = xs4 + s_ffn

        return out


class ChannelAttentionSegment(nn.Module):
    """
    Channel Attention Segment.

    Input shape:
        x: (B, C, T, F)

    Output shape:
        out: (B, C, T, F)
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        reduced_channels = max(1, in_channels // reduction)

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=in_channels
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=in_channels
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x) + self.conv2(x)

        # Global channel descriptor: (B, C, 1)
        gap = self.global_avg_pool(out).squeeze(-1).squeeze(-1).unsqueeze(-1)

        # Channel attention weights: (B, C, 1, 1)
        ca = self.attention(gap).unsqueeze(-1)

        out = out * ca
        out = self.conv_out(out)

        # Residual connection
        out = out + x

        return out


class TSAIC(nn.Module):
    """
    Temporal-Spectral Attention with Integrated Channel attention (TSAIC).

    The module combines:
        1. Temporal-Spectral Cross-Attention branch.
        2. Channel Attention branch.

    Input shape:
        x: (B, C, T, F)

    Output shape:
        out: (B, C, T, F)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int = 4,
        reduction: int = 16,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

        self.temporal_spectral = TemporalSpectralCrossAttentionSegment(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        self.channel_attention = ChannelAttentionSegment(
            in_channels=in_channels,
            reduction=reduction
        )

    def forward(self, x):
        temporal_spectral_features = self.temporal_spectral(x)
        channel_features = self.channel_attention(x)

        out = self.alpha * temporal_spectral_features + self.beta * channel_features

        return out
