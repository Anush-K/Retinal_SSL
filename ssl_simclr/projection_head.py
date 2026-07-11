"""
projection_head.py — Projection heads for LSFC pretraining

Adds TripleBranchProjectionHead (spatial + freq_fine + freq_coarse) for the
multi-band ablation, alongside the original single-head ProjectionHead
(spatial-only ablation) and DualBranchProjectionHead (single-band ablation,
kept unchanged for a faithful DBFC replication baseline).
"""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP: Linear -> BN -> ReLU -> Linear -> BN."""

    def __init__(self, in_dim: int = 1792, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DualBranchProjectionHead(nn.Module):
    """
    proj_spatial + proj_freq — kept unchanged from DBFC for the single-band
    ablation baseline (faithful replication of the deepfake design).
    """

    def __init__(self, in_dim: int = 1792, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj_spatial = ProjectionHead(in_dim, hidden_dim, out_dim)
        self.proj_freq    = ProjectionHead(in_dim, hidden_dim, out_dim)

    def forward_spatial(self, pooled_features):
        return self.proj_spatial(pooled_features)

    def forward_freq(self, pooled_features):
        return self.proj_freq(pooled_features)


class TripleBranchProjectionHead(nn.Module):
    """
    proj_spatial + proj_freq_fine + proj_freq_coarse — the proposed LSFC head.

    proj_freq_fine   : aligned with microaneurysm-scale frequency view
    proj_freq_coarse : aligned with hemorrhage/exudate-scale frequency view

    Independent heads let each frequency scale specialize, rather than
    forcing both lesion scales into a single shared subspace.
    """

    def __init__(self, in_dim: int = 1792, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj_spatial    = ProjectionHead(in_dim, hidden_dim, out_dim)
        self.proj_freq_fine  = ProjectionHead(in_dim, hidden_dim, out_dim)
        self.proj_freq_coarse = ProjectionHead(in_dim, hidden_dim, out_dim)

    def forward_spatial(self, pooled_features):
        return self.proj_spatial(pooled_features)

    def forward_freq_fine(self, pooled_features):
        return self.proj_freq_fine(pooled_features)

    def forward_freq_coarse(self, pooled_features):
        return self.proj_freq_coarse(pooled_features)