"""Conformer blocks for the Raw-SEEG state encoder (revision R0.2).

Why this exists
---------------
R0.1's encoder is a pure Transformer: three ``nn.TransformerEncoder`` stacks
with a Conv1d stack used only as the patch tokeniser at the front. Nothing
inside a block is convolutional. On the first subject that finished every
control arm, that encoder lost to a 1008-coefficient ridge on spectral history
at all four horizons, and the 60-epoch arm converged (validation rose after
epoch 56), so "train it longer" is not the explanation. What is left on the
table is a representation bottleneck, and the most suspicious place is the
temporal stage: it has to read local waveform morphology -- spikes, oscillatory
bursts, sharp transients -- out of twenty 250 ms patches using nothing but
self-attention over twenty tokens. Self-attention is a poor tool for that;
depthwise convolution is the right one.

A Conformer block interleaves both:

    x = x + 1/2 FFN(x)
    x = x + MHSA(LN(x))
    x = x + Conv(LN(x))          <- the part R0.1 has nowhere
    x = x + 1/2 FFN(x)
    x = LN(x)

The convolution module is the standard one: pointwise conv expanding 2x, GLU,
depthwise conv, normalisation, Swish, pointwise conv back.

Causality
---------
The context stage runs over minute tokens and **must stay causal** -- minute t
may not see minute t+1, or the whole open-loop premise is void. A symmetric
depthwise convolution would leak the future silently, which is exactly the kind
of defect that produces a beautiful and worthless result. ``causal=True`` pads
only on the left and trims the right, and ``tests/test_raw_seeg_state_conformer.py``
checks it by perturbation, not by reading the code.

The temporal stage is not causal and does not need to be: the whole 5 s window
is already in the past at prediction time.

The spatial stage keeps a plain Transformer. Its axis is contacts, not time; a
depthwise convolution there would impose a neighbour relation that only holds
inside a shaft and silently crosses shaft boundaries. One change at a time, on
the axis where the inductive bias is unambiguous.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class ConvModule(nn.Module):
    """Pointwise -> GLU -> depthwise -> norm -> Swish -> pointwise.

    ``causal=True`` left-pads by ``kernel_size - 1`` and trims the tail, so
    position t is a function of positions <= t only.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if not causal and kernel_size % 2 == 0:
            raise ValueError("a non-causal kernel must be odd so it stays centred")
        self.causal = bool(causal)
        self.kernel_size = int(kernel_size)
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=self.kernel_size,
            groups=d_model,
            padding=0 if self.causal else self.kernel_size // 2,
        )
        # LayerNorm, not BatchNorm: batches here are small (1-8 windows) and a
        # BatchNorm would make one sample's representation depend on the others
        # in its batch, which is not something a per-patient state model should
        # ever do.
        self.post_norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:            # (B, T, D) -> (B, T, D)
        y = self.norm(x).transpose(1, 2)               # (B, D, T)
        y = nn.functional.glu(self.pointwise_in(y), dim=1)
        if self.causal:
            y = nn.functional.pad(y, (self.kernel_size - 1, 0))
        y = self.depthwise(y)
        y = self.post_norm(y.transpose(1, 2))          # (B, T, D)
        y = self.act(y)
        y = self.pointwise_out(y.transpose(1, 2)).transpose(1, 2)
        return self.dropout(y)


class _HalfFFN(nn.Module):
    """Pre-LN feed-forward with the Conformer's 1/2 residual weight."""

    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * self.net(x)


class ConformerBlock(nn.Module):
    """Half-FFN, self-attention, convolution, half-FFN, final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 dropout: float = 0.1, kernel_size: int = 7, causal: bool = False):
        super().__init__()
        self.ffn1 = _HalfFFN(d_model, ffn_mult, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, kernel_size=kernel_size, dropout=dropout,
                               causal=causal)
        self.ffn2 = _HalfFFN(d_model, ffn_mult, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.ffn1(x)
        h = self.attn_norm(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.attn_drop(a)
        x = x + self.conv(x)
        x = x + self.ffn2(x)
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    """A stack of :class:`ConformerBlock`, drop-in for nn.TransformerEncoder.

    Accepts the same ``mask`` / ``src_key_padding_mask`` / ``is_causal``
    keywords the Transformer call sites already pass, so the encoder kind can be
    switched without touching those call sites.
    """

    def __init__(self, n_layers: int, d_model: int, n_heads: int, ffn_mult: int = 4,
                 dropout: float = 0.1, kernel_size: int = 7, causal: bool = False):
        super().__init__()
        self.causal = bool(causal)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ffn_mult, dropout, kernel_size, causal)
            for _ in range(int(n_layers))
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: Optional[bool] = None) -> Tensor:
        if mask is not None and not self.causal:
            # A causal attention mask on a stack whose convolution is symmetric
            # would be a lie: the conv would carry the future around the mask.
            raise ValueError(
                "an attention mask was supplied to a non-causal ConformerEncoder; "
                "build it with causal=True so the depthwise convolution is causal too"
            )
        for layer in self.layers:
            x = layer(x, attn_mask=mask, key_padding_mask=src_key_padding_mask)
        return x
