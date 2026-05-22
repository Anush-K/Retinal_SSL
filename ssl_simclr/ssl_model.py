"""
ssl_model.py — DBFC SSL model

Changes from original:
- Original: SSLModel with single ProjectionHead, forward(x) → z (one embedding)
- New: SSLModel with DualBranchProjectionHead.
  forward() now accepts view type and routes through the correct projection head.
  Three forward calls per training step:
    forward(view_s1, domain='spatial') → z_s1
    forward(view_s2, domain='spatial') → z_s2
    forward(view_f,  domain='freq')   → z_f

  The backbone and GeM pool are SHARED across all views — only the final
  projection layer differs per domain. This keeps compute nearly identical
  to the original (one extra projection head = ~200K params, negligible).

NOTE FOR FINE-TUNING COMPATIBILITY:
  The backbone and pool keys in the saved state dict are IDENTICAL to the
  original ('backbone.*', 'pool.*'). The fine-tuning scripts
  (train_finetune.py, train_finetune_no_ssl.py, etc.) load these keys
  with the same logic as before — ZERO changes required to those files.
  The 'projector.*' keys are simply ignored during fine-tuning load.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ssl_simclr.gem import GeM
from ssl_simclr.projection_head import DualBranchProjectionHead


class SSLModel(nn.Module):
    """
    Dual-Branch Frequency Contrastive (DBFC) SSL model.

    Architecture:
        Input image → EfficientNet-B4 (backbone) → GeM pool → [1792-dim pooled features]
            ├─ proj_spatial → [128-dim z_s]  (for spatial views)
            └─ proj_freq    → [128-dim z_f]  (for frequency views)

    The backbone is shared; only the projection heads branch.
    All embeddings are L2-normalized before contrastive loss.
    """

    def __init__(self):
        super().__init__()

        # EfficientNet-B4: final_conv output is [B, 1792, H, W]
        # num_classes=0 removes the original classifier head
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,
        )

        # GeM pooling: learnable p parameter, more selective than average pooling
        # Preserved for compatibility with fine-tuning scripts
        self.pool = GeM()

        # Dual-branch projection heads (discarded after pretraining)
        # Named 'projector' for state dict compatibility — fine-tune scripts
        # already filter out any key not starting with 'backbone.' or 'pool.'
        self.projector = DualBranchProjectionHead(
            in_dim=1792, hidden_dim=512, out_dim=128
        )

    def encode(self, x):
        """Shared encoder: backbone + GeM pool → pooled features [B, 1792]."""
        feat_map = self.backbone.forward_features(x)   # [B, 1792, 7, 7]
        pooled   = self.pool(feat_map).flatten(1)       # [B, 1792]
        return pooled

    def forward(self, x, domain: str = 'spatial'):
        """
        Args:
            x      : input image tensor [B, 3, 224, 224]
            domain : 'spatial' or 'freq' — selects which projection head to use

        Returns:
            z : L2-normalized embedding [B, 128]
        """
        pooled = self.encode(x)

        if domain == 'spatial':
            z = self.projector.forward_spatial(pooled)
        elif domain == 'freq':
            z = self.projector.forward_freq(pooled)
        else:
            raise ValueError(f"Unknown domain '{domain}'. Must be 'spatial' or 'freq'.")

        return F.normalize(z, dim=1)