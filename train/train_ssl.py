"""
train_ssl.py — LSFC SSL pretraining, 4-way mode support.

  --mode spatial_only  : plain SimCLR (no frequency branch)
  --mode single_band    : DBFC replica (single kernel=5 high-pass)
  --mode multi_band     : naive multi-band (fine+coarse share global pool —
                           produced redundant fine/coarse embeddings,
                           cosine sim ~0.96, in the original experiment)
  --mode multi_band_sp  : FIX — fine/coarse pooled at higher spatial
                           resolution before flattening (scale-preserving)

Usage (from Colab):
    !PYTHONPATH=/content/Retinal_SSL python train/train_ssl.py --mode multi_band_sp
"""

import os
import glob
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ssl_simclr.ssl_dataset import SSLDataset
from ssl_simclr.ssl_model import SSLModel
from ssl_simclr.contrastive_loss import dbfc_loss, lsfc_loss, nt_xent_loss

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True,
                    choices=["spatial_only", "single_band", "multi_band", "multi_band_sp"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--csv_files", nargs="+",
                    default=["/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv"])
    return p.parse_args()


def train_ssl(mode, csv_files, device="cuda", epochs=50, batch_size=64, lr=3e-4):
    print(f"[{mode}] Training started: "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    dataset = SSLDataset(csv_files)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8,
        pin_memory=True, persistent_workers=True, drop_last=True,
    )
    print(f"Dataset size : {len(dataset)} | Batches/epoch: {len(loader)}")

    model = SSLModel(mode=mode).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint_dir = f"/content/drive/MyDrive/Retinal_SSL/checkpoints_{mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/ssl_epoch_*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Resuming from: {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}")

    USES_FREQ_FINE_COARSE = mode in ("multi_band", "multi_band_sp")

    for epoch in range(start_epoch, epochs):
        model.train()
        totals = {"loss": 0.0, "within": 0.0, "fine": 0.0, "coarse": 0.0}

        for view_s1, view_s2, view_fine, view_coarse, view_legacy in tqdm(
            loader, desc=f"[{mode}] Epoch {epoch+1}/{epochs}"
        ):
            view_s1 = view_s1.to(device, non_blocking=True)
            view_s2 = view_s2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                z_s1 = model(view_s1, domain="spatial")
                z_s2 = model(view_s2, domain="spatial")

                if mode == "spatial_only":
                    loss = nt_xent_loss(z_s1, z_s2, temperature=0.1)
                    l_w, l_fine, l_coarse = loss.item(), 0.0, 0.0

                elif mode == "single_band":
                    view_legacy = view_legacy.to(device, non_blocking=True)
                    z_f = model(view_legacy, domain="freq")
                    loss = dbfc_loss(z_s1, z_s2, z_f)
                    with torch.no_grad():
                        l_w = nt_xent_loss(z_s1, z_s2, temperature=0.1).item()
                    l_fine, l_coarse = 0.0, 0.0

                else:  # multi_band or multi_band_sp — identical training loop,
                       # only the model's internal pooling differs
                    view_fine   = view_fine.to(device, non_blocking=True)
                    view_coarse = view_coarse.to(device, non_blocking=True)
                    z_fine   = model(view_fine,   domain="freq_fine")
                    z_coarse = model(view_coarse, domain="freq_coarse")
                    loss, l_w_t, l_fine_t, l_coarse_t = lsfc_loss(
                        z_s1, z_s2, z_fine, z_coarse
                    )
                    l_w, l_fine, l_coarse = (
                        l_w_t.item(), l_fine_t.item(), l_coarse_t.item()
                    )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            totals["loss"]   += loss.item()
            totals["within"] += l_w
            totals["fine"]   += l_fine
            totals["coarse"] += l_coarse

        scheduler.step()
        n = len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{mode}] Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {totals['loss']/n:.4f} | "
            f"L_within: {totals['within']/n:.4f} | "
            f"L_fine: {totals['fine']/n:.4f} | "
            f"L_coarse: {totals['coarse']/n:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if (epoch + 1) % 5 == 0:
            ckpt_path = f"{checkpoint_dir}/ssl_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    final_path = f"{checkpoint_dir}/ssl_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[{mode}] SSL pretraining complete. Saved: {final_path}")
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    args = parse_args()
    train_ssl(
        mode=args.mode,
        csv_files=args.csv_files,
        device="cuda",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )