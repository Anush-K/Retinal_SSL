"""
projection_head.py — Projection heads for DBFC pretraining

Changes from original:
- Original: single ProjectionHead (1792 → 512 → 128)
- New: DualBranchProjectionHead containing two independent heads:
    * proj_spatial : maps backbone features from spatial views
    * proj_freq    : maps backbone features from frequency-enhanced views

  Both heads have identical architecture (same as original) but separate
  weights, so they can specialize to their respective domains.

  WHY two heads:
  A single shared head would force spatial and frequency embeddings into
  the same linear subspace from the very first layer. Two separate heads
  let the model learn domain-specific linear transformations before the
  representations are compared in contrastive loss. This is analogous to
  how CLIP uses separate image and text encoders before a shared embedding.

  After pretraining, BOTH heads are discarded. Only backbone + GeM are
  transferred to fine-tuning (same as original workflow).
"""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Single two-layer MLP projection head.
    Architecture: Linear → BN → ReLU → Linear → BN
    L2 normalization applied OUTSIDE this module (in ssl_model.py forward).
    """

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
    Two independent projection heads sharing the same architecture.

    proj_spatial : used for spatial views (view_s1, view_s2)
    proj_freq    : used for frequency-enhanced views (view_f)

    Args:
        in_dim     : backbone output dimension (1792 for EfficientNet-B4 + GeM)
        hidden_dim : intermediate MLP dimension (512)
        out_dim    : final embedding dimension (128)
    """

    def __init__(self, in_dim: int = 1792, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj_spatial = ProjectionHead(in_dim, hidden_dim, out_dim)
        self.proj_freq    = ProjectionHead(in_dim, hidden_dim, out_dim)

    def forward_spatial(self, pooled_features):
        return self.proj_spatial(pooled_features)

    def forward_freq(self, pooled_features):
        return self.proj_freq(pooled_features)