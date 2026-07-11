"""
contrastive_loss.py — LSFC contrastive losses

nt_xent_loss : unchanged, standard SimCLR loss — used directly for
               spatial_only mode.
dbfc_loss    : unchanged from the deepfake project — used for the
               single_band ablation (faithful DBFC replica).
lsfc_loss    : NEW — three-term loss for the multi_band (proposed) mode.
               The frequency budget (0.3, same total as DBFC) is split
               evenly across the two lesion scales rather than
               concentrated in one band, so the total spatial/frequency
               balance is preserved for a fair comparison against DBFC.
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t())
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))
    sim = sim / temperature
    targets = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, targets)


def dbfc_loss(
    z_s1: torch.Tensor, z_s2: torch.Tensor, z_f: torch.Tensor,
    tau_within: float = 0.1, tau_cross: float = 0.2,
    lambda_within: float = 0.7, lambda_cross: float = 0.3,
) -> torch.Tensor:
    """Unchanged from the deepfake project — used for single_band ablation."""
    L_within = nt_xent_loss(z_s1, z_s2, temperature=tau_within)
    L_cross  = nt_xent_loss(z_s1, z_f,  temperature=tau_cross)
    return lambda_within * L_within + lambda_cross * L_cross


def lsfc_loss(
    z_s1: torch.Tensor,
    z_s2: torch.Tensor,
    z_fine: torch.Tensor,
    z_coarse: torch.Tensor,
    tau_within: float = 0.1,
    tau_fine: float = 0.2,
    tau_coarse: float = 0.2,
    lambda_within: float = 0.7,
    lambda_fine: float = 0.15,
    lambda_coarse: float = 0.15,
):
    """
    Lesion-Scale Frequency Contrastive (LSFC) loss.

    L = 0.7 * L_within(s1,s2)
      + 0.15 * L_fine(s1, f_fine)      [microaneurysm-scale alignment]
      + 0.15 * L_coarse(s1, f_coarse)  [hemorrhage/exudate-scale alignment]

    Returns (total_loss, L_within, L_fine, L_coarse) so all components can
    be logged separately during training.
    """
    L_within = nt_xent_loss(z_s1, z_s2,     temperature=tau_within)
    L_fine   = nt_xent_loss(z_s1, z_fine,   temperature=tau_fine)
    L_coarse = nt_xent_loss(z_s1, z_coarse, temperature=tau_coarse)

    total = (lambda_within * L_within
             + lambda_fine * L_fine
             + lambda_coarse * L_coarse)

    return total, L_within, L_fine, L_coarse