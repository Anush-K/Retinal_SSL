"""
projection_head.py — LSFC projection heads, now including the
scale-preserving variant (multi_band_sp).
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
    """proj_spatial + proj_freq — single_band ablation (DBFC replica)."""

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
    proj_spatial + proj_freq_fine + proj_freq_coarse — original multi_band.
    All three take the SAME 1792-d globally-pooled input (this is the
    architecture that produced the redundant fine/coarse embeddings).
    """

    def __init__(self, in_dim: int = 1792, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj_spatial     = ProjectionHead(in_dim, hidden_dim, out_dim)
        self.proj_freq_fine   = ProjectionHead(in_dim, hidden_dim, out_dim)
        self.proj_freq_coarse = ProjectionHead(in_dim, hidden_dim, out_dim)

    def forward_spatial(self, pooled_features):
        return self.proj_spatial(pooled_features)

    def forward_freq_fine(self, pooled_features):
        return self.proj_freq_fine(pooled_features)

    def forward_freq_coarse(self, pooled_features):
        return self.proj_freq_coarse(pooled_features)


class ScaleAwareTripleBranchProjectionHead(nn.Module):
    """
    multi_band_sp (scale-preserving) — proj_spatial takes the global 1792-d
    vector as before, but proj_freq_fine and proj_freq_coarse take HIGHER-
    DIMENSIONAL, spatially-structured inputs (flattened multi-region GeM
    grids), so scale/layout information is available for them to
    differentiate on, unlike the original TripleBranchProjectionHead.
    """

    def __init__(self, spatial_in_dim: int, fine_in_dim: int, coarse_in_dim: int,
                 hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj_spatial     = ProjectionHead(spatial_in_dim, hidden_dim, out_dim)
        self.proj_freq_fine   = ProjectionHead(fine_in_dim,   hidden_dim, out_dim)
        self.proj_freq_coarse = ProjectionHead(coarse_in_dim, hidden_dim, out_dim)

    def forward_spatial(self, pooled_features):
        return self.proj_spatial(pooled_features)

    def forward_freq_fine(self, pooled_features):
        return self.proj_freq_fine(pooled_features)

    def forward_freq_coarse(self, pooled_features):
        return self.proj_freq_coarse(pooled_features)