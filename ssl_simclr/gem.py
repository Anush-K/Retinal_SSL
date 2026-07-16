import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Original global GeM pool — unchanged, used for the spatial branch
    and kept identical for backbone/pool checkpoint compatibility."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class MultiRegionGeM(nn.Module):
    """
    Scale-preserving GeM pool — pools to a small spatial grid instead of a
    single global vector, so spatial/scale layout survives into the
    projection head.

    Used ONLY for the frequency branches in multi_band_sp mode. The spatial
    branch continues to use the original global GeM above, so backbone.*/
    pool.* checkpoint keys used by train_finetune.py are unaffected —
    this module is registered under a different name (freq_pool_fine /
    freq_pool_coarse) and is simply ignored (never loaded, never expected)
    during fine-tuning.

    output_size=(1,1) is mathematically identical to GeM above (a sanity
    check you can use if needed).
    """

    def __init__(self, output_size=(2, 2), p=3.0, eps=1e-6):
        super().__init__()
        self.output_size = output_size
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), self.output_size
        ).pow(1.0 / self.p)