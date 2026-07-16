"""
ssl_model.py — LSFC SSL model, 4-way mode support.

mode='spatial_only' : ProjectionHead only              — plain SimCLR
mode='single_band'   : DualBranchProjectionHead          — DBFC replica
mode='multi_band'    : TripleBranchProjectionHead        — naive multi-band
                        (fine/coarse share the same globally-pooled 1792-d
                        input — this is the version that showed fine/coarse
                        embedding redundancy, cosine sim ~0.96)
mode='multi_band_sp' : ScaleAwareTripleBranchProjectionHead — FIX: fine and
                        coarse frequency branches are pooled at higher
                        spatial resolution (3x3 and 2x2 grids respectively)
                        BEFORE flattening, so scale/layout information
                        survives into the projection heads. Spatial branch
                        pooling is untouched (still global GeM), so
                        backbone.*/pool.* checkpoint keys remain identical
                        for train_finetune.py compatibility.
"""

import torch.nn as nn
import torch.nn.functional as F
import timm

from ssl_simclr.gem import GeM, MultiRegionGeM
from ssl_simclr.projection_head import (
    ProjectionHead, DualBranchProjectionHead, TripleBranchProjectionHead,
    ScaleAwareTripleBranchProjectionHead
)


class SSLModel(nn.Module):
    def __init__(self, mode: str = "multi_band_sp",
                 fine_grid=(3, 3), coarse_grid=(2, 2)):
        super().__init__()
        assert mode in ("spatial_only", "single_band", "multi_band", "multi_band_sp")
        self.mode = mode

        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0
        )
        self.pool = GeM()  # global pool — used for spatial branch in ALL modes

        BACKBONE_CH = 1792

        if mode == "spatial_only":
            self.projector = ProjectionHead(in_dim=BACKBONE_CH, hidden_dim=512, out_dim=128)

        elif mode == "single_band":
            self.projector = DualBranchProjectionHead(in_dim=BACKBONE_CH, hidden_dim=512, out_dim=128)

        elif mode == "multi_band":
            self.projector = TripleBranchProjectionHead(in_dim=BACKBONE_CH, hidden_dim=512, out_dim=128)

        else:  # multi_band_sp
            self.fine_grid = fine_grid
            self.coarse_grid = coarse_grid
            self.freq_pool_fine   = MultiRegionGeM(output_size=fine_grid)
            self.freq_pool_coarse = MultiRegionGeM(output_size=coarse_grid)

            fine_in_dim   = BACKBONE_CH * fine_grid[0]   * fine_grid[1]
            coarse_in_dim = BACKBONE_CH * coarse_grid[0] * coarse_grid[1]

            self.projector = ScaleAwareTripleBranchProjectionHead(
                spatial_in_dim=BACKBONE_CH,
                fine_in_dim=fine_in_dim,
                coarse_in_dim=coarse_in_dim,
                hidden_dim=512, out_dim=128
            )

    def forward(self, x, domain: str = "spatial"):
        feat_map = self.backbone.forward_features(x)  # [B, 1792, 7, 7]

        if self.mode == "spatial_only":
            pooled = self.pool(feat_map).flatten(1)
            z = self.projector(pooled)

        elif self.mode == "single_band":
            if domain == "spatial":
                pooled = self.pool(feat_map).flatten(1)
                z = self.projector.forward_spatial(pooled)
            elif domain == "freq":
                pooled = self.pool(feat_map).flatten(1)
                z = self.projector.forward_freq(pooled)
            else:
                raise ValueError(f"single_band: unknown domain '{domain}'")

        elif self.mode == "multi_band":
            pooled = self.pool(feat_map).flatten(1)  # SAME pooled vector for all branches
            if domain == "spatial":
                z = self.projector.forward_spatial(pooled)
            elif domain == "freq_fine":
                z = self.projector.forward_freq_fine(pooled)
            elif domain == "freq_coarse":
                z = self.projector.forward_freq_coarse(pooled)
            else:
                raise ValueError(f"multi_band: unknown domain '{domain}'")

        else:  # multi_band_sp
            if domain == "spatial":
                pooled = self.pool(feat_map).flatten(1)
                z = self.projector.forward_spatial(pooled)
            elif domain == "freq_fine":
                pooled = self.freq_pool_fine(feat_map).flatten(1)   # [B, 1792*9]
                z = self.projector.forward_freq_fine(pooled)
            elif domain == "freq_coarse":
                pooled = self.freq_pool_coarse(feat_map).flatten(1)  # [B, 1792*4]
                z = self.projector.forward_freq_coarse(pooled)
            else:
                raise ValueError(f"multi_band_sp: unknown domain '{domain}'")

        return F.normalize(z, dim=1)