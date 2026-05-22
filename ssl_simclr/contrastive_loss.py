"""
contrastive_loss.py — DBFC contrastive loss

Changes from original:
- Original: nt_xent_loss(z1, z2, temperature=0.1) — single NT-Xent
- New: dbfc_loss(z_s1, z_s2, z_f) — two-component loss:
    1. L_within : NT-Xent between two spatial views (τ=0.1)
    2. L_cross  : NT-Xent between spatial and frequency views (τ=0.2)
    Total = λ_within * L_within + λ_cross * L_cross

Original nt_xent_loss() is PRESERVED for backward compatibility and ablation.

Why different temperatures:
- τ=0.1 (within-domain): spatial views are augmentations of the same RGB image.
  They should be very similar in feature space. Low temperature creates sharper
  peaks → stronger learning signal for clear positives.
- τ=0.2 (cross-domain): spatial and frequency views capture genuinely different
  signal (color vs edge residuals). Forcing them to be identical (low τ) would
  cause representation collapse or conflicting gradients. A softer temperature
  allows the model to align their abstract manipulation-sensitive content
  WITHOUT forcing pixel-level correspondence.

Why λ_within=0.7, λ_cross=0.3:
- Within-domain loss is the primary objective (proven to work from SimCLR).
- Cross-domain alignment is a regularizer that steers representations toward
  frequency-sensitive features. Over-weighting it risks degrading spatial
  representations that fine-tuning depends on.
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy loss (NT-Xent).

    Standard SimCLR loss. Preserved from original for ablation experiments.

    Args:
        z1, z2      : L2-normalized embeddings [B, D]
        temperature : softmax temperature τ

    Returns:
        scalar loss
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # [2B, D]

    # Cosine similarity matrix (z is already L2-normalized, so mm = cosine sim)
    sim = torch.mm(z, z.t())                 # [2B, 2B]

    # Mask self-similarity before scaling (avoids -inf dominating softmax)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    sim = sim / temperature

    # Positive pairs: (i, i+B) and (i+B, i)
    targets = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])

    return F.cross_entropy(sim, targets)


def dbfc_loss(
    z_s1: torch.Tensor,
    z_s2: torch.Tensor,
    z_f:  torch.Tensor,
    tau_within: float = 0.1,
    tau_cross:  float = 0.2,
    lambda_within: float = 0.7,
    lambda_cross:  float = 0.3,
) -> torch.Tensor:
    """
    Dual-Branch Frequency Contrastive (DBFC) loss.

    Two components:
    1. Within-domain spatial loss (primary):
       NT-Xent(z_s1, z_s2) — two independently augmented spatial views
       of the same image should have similar embeddings.

    2. Cross-domain alignment loss (regularizer):
       NT-Xent(z_s1, z_f) — spatial and frequency views of the same image
       should be more similar to each other than to views of other images,
       but NOT forced to be identical (softer temperature).

    Args:
        z_s1, z_s2 : L2-normalized spatial embeddings [B, 128]
        z_f        : L2-normalized frequency embeddings [B, 128]
        tau_within : temperature for within-domain loss (0.1, sharper)
        tau_cross  : temperature for cross-domain loss (0.2, softer)
        lambda_within : weight for within-domain loss (0.7)
        lambda_cross  : weight for cross-domain loss (0.3)

    Returns:
        total scalar loss

    NOTE: z_s1, z_s2 are produced by proj_spatial (same head).
          z_f is produced by proj_freq (separate head).
          Both are L2-normalized in ssl_model.forward().
    """
    L_within = nt_xent_loss(z_s1, z_s2, temperature=tau_within)
    L_cross  = nt_xent_loss(z_s1, z_f,  temperature=tau_cross)

    return lambda_within * L_within + lambda_cross * L_cross