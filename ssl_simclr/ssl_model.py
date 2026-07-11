"""
ssl_model.py — LSFC SSL model, mode-aware for the 3-way ablation.

mode='spatial_only' : ProjectionHead only              — plain SimCLR baseline
mode='single_band'   : DualBranchProjectionHead          — faithful DBFC replica
mode='multi_band'    : TripleBranchProjectionHead (ours) — proposed LSFC

Backbone + GeM pool are IDENTICAL across all three modes and IDENTICAL in
key-naming to the deepfake project's checkpoints ('backbone.*', 'pool.*'),
so train_finetune.py needs zero changes to load any of these checkpoints.
"""

import torch.nn as nn
import torch.nn.functional as F
import timm

from ssl_simclr.gem import GeM
from ssl_simclr.projection_head import (
    ProjectionHead, DualBranchProjectionHead, TripleBranchProjectionHead
)


class SSLModel(nn.Module):
    def __init__(self, mode: str = "multi_band"):
        super().__init__()
        assert mode in ("spatial_only", "single_band", "multi_band")
        self.mode = mode

        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0
        )
        self.pool = GeM()

        if mode == "spatial_only":
            self.projector = ProjectionHead(in_dim=1792, hidden_dim=512, out_dim=128)
        elif mode == "single_band":
            self.projector = DualBranchProjectionHead(in_dim=1792, hidden_dim=512, out_dim=128)
        else:  # multi_band
            self.projector = TripleBranchProjectionHead(in_dim=1792, hidden_dim=512, out_dim=128)

    def encode(self, x):
        feat_map = self.backbone.forward_features(x)
        return self.pool(feat_map).flatten(1)

    def forward(self, x, domain: str = "spatial"):
        pooled = self.encode(x)

        if self.mode == "spatial_only":
            z = self.projector(pooled)
        elif self.mode == "single_band":
            if domain == "spatial":
                z = self.projector.forward_spatial(pooled)
            elif domain == "freq":
                z = self.projector.forward_freq(pooled)
            else:
                raise ValueError(f"single_band mode: unknown domain '{domain}'")
        else:  # multi_band
            if domain == "spatial":
                z = self.projector.forward_spatial(pooled)
            elif domain == "freq_fine":
                z = self.projector.forward_freq_fine(pooled)
            elif domain == "freq_coarse":
                z = self.projector.forward_freq_coarse(pooled)
            else:
                raise ValueError(f"multi_band mode: unknown domain '{domain}'")

        return F.normalize(z, dim=1)