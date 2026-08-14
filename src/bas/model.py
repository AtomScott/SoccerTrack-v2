"""A dilated temporal convolutional spotter over trajectory features.

Per docs/experiment-design-bas.md section 3: "a temporal model over the per-frame feature
sequence -- a TCN or a small transformer -- predicting per-frame class logits, decoded to
spots by peak-picking with non-maximum suppression."

A TCN rather than a transformer because the useful context is local and the sequences are
long: at the default 5 Hz a half is ~14,000 steps, and six residual blocks with dilations
1..32 already see +/-63 steps (25 s) either side, which comfortably covers the build-up to
a ball event. It also trains on CPU in minutes, which is the operative constraint while the
GPU is committed to the GSR runs.

Twelve INDEPENDENT sigmoid outputs, not a softmax over classes plus background. Ball events
co-occur by design -- a goal is annotated as a Shot and a Goal at the same timestamp, and
docs/format-bas.md says not to collapse them -- so a softmax would force the model to
choose between two labels that are both correct.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Dilated conv -> norm -> GELU -> dropout, twice, plus a residual connection."""

    def __init__(self, ch: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        # GroupNorm, not BatchNorm: batches are few long sequences rather than many short
        # ones, so batch statistics are dominated by which halves happen to be in the batch.
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return x + h


class TrajectorySpotter(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 12, hidden: int = 128,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32), dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_features, hidden, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, d, dropout=dropout)
                                      for d in dilations])
        self.head = nn.Conv1d(hidden, n_classes, 1)
        # Start with a low prior probability on every class. Ball events are sparse in time
        # (~1 positive row in 25 per class at 5 Hz for the commonest class, far fewer for
        # the rest); without this the first hundred steps are spent unlearning p ~ 0.5.
        nn.init.constant_(self.head.bias, -4.0)
        self.receptive_field = 1 + 4 * sum(dilations)

    def forward(self, x):
        """x: (B, T, F) -> logits (B, T, C)."""
        h = self.inp(x.transpose(1, 2))
        h = self.blocks(h)
        return self.head(h).transpose(1, 2)
