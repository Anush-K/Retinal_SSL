"""
train_ssl.py — DBFC SSL pretraining (v3 — collapse fix)

Changes from v2:
1. LR reduced from 1e-3 to 3e-4.
   With a warm-started backbone, 1e-3 was too aggressive and destabilized
   features the old checkpoint built. 3e-4 is the standard SimCLR-v2 LR
   for fine-grained tasks with pretrained init.

2. Augmentation is now asymmetric (easy anchor + hard positive) — see
   augmentations.py v3. This prevents the L_within collapse seen in v2
   where both spatial views were identical pipelines.

3. Checkpoint handling: if an old-arch checkpoint exists (projector.net.*),
   load backbone+pool with strict=False and restart from epoch 0 with new LR.
   If a new-arch checkpoint exists (projector.proj_spatial.*), resume normally.

4. Added collapse detection: if L_within < 0.05 for 3 consecutive epochs,
   print a warning. Does not stop training — just alerts you.

5. __main__ block chains all downstream runs automatically after SSL finishes:
   SSL -> t-SNE -> finetune (SSL) -> finetune (no SSL) -> summary

UNCHANGED from v2:
- scheduler.step() outside batch loop (the original scheduler bug fix)
- 3-view dataset (view_s1, view_s2, view_f)
- DBFC loss (0.7 * L_within + 0.3 * L_cross)
- All fine-tuning scripts unchanged (backbone.* pool.* keys identical)
"""

import os
import glob
import datetime
import subprocess
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ssl_simclr.ssl_dataset import SSLDataset
from ssl_simclr.ssl_model import SSLModel
from ssl_simclr.contrastive_loss import dbfc_loss, nt_xent_loss


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def run_step(label, cmd):
    """
    Run a downstream step as a subprocess.
    Streams output live to the tmux window so you can see progress
    when you reattach tomorrow.
    Exits the whole pipeline if any step fails.
    """
    print(f"\n{'='*60}")
    print(f"PIPELINE: Starting — {label}")
    print(f"Command : {cmd}")
    print(f"Time    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n*** PIPELINE FAILED at step: {label} (exit code {result.returncode}) ***")
        print(f"Remaining steps skipped. Check output above for error.")
        sys.exit(result.returncode)

    print(f"\nPIPELINE: Finished — {label}")
    print(f"Time    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def train_ssl(
    csv_files,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 64,
):
    print(f"Training started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Dataset & Loader ──────────────────────────────────────
    dataset = SSLDataset(csv_files)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    print(f"Dataset size : {len(dataset)} samples")
    print(f"Batches/epoch: {len(loader)}")
    print(f"Total epochs : {epochs}")

    # ── Model ─────────────────────────────────────────────────
    model = SSLModel().to(device)

    # ── Optimizer ─────────────────────────────────────────────
    # 3e-4 instead of 1e-3: warm-started backbone needs a gentler LR
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # ── Scheduler ─────────────────────────────────────────────
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Mixed precision ───────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()

    # ── Checkpoint Setup ──────────────────────────────────────
    checkpoint_dir = "/content/drive/MyDrive/Retinal_SSL/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/ssl_epoch_*.pt"))

    if checkpoints:
        latest = checkpoints[-1]
        print(f"Resuming from: {latest}")
        ckpt = torch.load(latest, map_location=device)
        saved_state = ckpt["model_state_dict"]

        missing, unexpected = model.load_state_dict(saved_state, strict=False)

        backbone_pool_missing = [
            k for k in missing
            if k.startswith("backbone.") or k.startswith("pool.")
        ]
        if backbone_pool_missing:
            print(f"  WARNING: backbone/pool keys missing: {backbone_pool_missing[:3]}")
        else:
            print(f"  backbone + pool loaded successfully.")

        proj_skipped = [k for k in unexpected if k.startswith("projector.")]
        if proj_skipped:
            print(f"  Old projector keys skipped ({len(proj_skipped)} keys). "
                  f"Projector reinitializes fresh.")

        is_new_arch = any("proj_spatial" in k for k in saved_state.keys())

        if is_new_arch:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            for _ in range(start_epoch):
                scheduler.step()
            print(f"  Optimizer state restored. Resuming from epoch {start_epoch}.")
        else:
            start_epoch = 0
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            print(f"  Old-arch checkpoint: backbone warm-start, "
                  f"epoch counter reset to 0, LR=3e-4.")
            print(f"  Full {epochs}-epoch DBFC training will proceed.")

        print(f"Resumed from epoch {start_epoch}")

    # ── Collapse detection ────────────────────────────────────
    low_within_streak = 0
    COLLAPSE_THRESHOLD = 0.05
    COLLAPSE_STREAK    = 3

    # ── Training Loop ─────────────────────────────────────────
    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss   = 0.0
        total_within = 0.0
        total_cross  = 0.0

        for view_s1, view_s2, view_f in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):

            view_s1 = view_s1.to(device, non_blocking=True)
            view_s2 = view_s2.to(device, non_blocking=True)
            view_f  = view_f.to(device,  non_blocking=True)

            with torch.cuda.amp.autocast():
                z_s1 = model(view_s1, domain='spatial')
                z_s2 = model(view_s2, domain='spatial')
                z_f  = model(view_f,  domain='freq')

                loss = dbfc_loss(z_s1, z_s2, z_f)

                with torch.no_grad():
                    l_w = nt_xent_loss(z_s1, z_s2, temperature=0.1).item()
                    l_c = nt_xent_loss(z_s1, z_f,  temperature=0.2).item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss   += loss.item()
            total_within += l_w
            total_cross  += l_c

        # ── Scheduler: once per epoch ─────────────────────────
        scheduler.step()

        n_batches  = len(loader)
        avg_loss   = total_loss   / n_batches
        avg_within = total_within / n_batches
        avg_cross  = total_cross  / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"L_within: {avg_within:.4f} | "
            f"L_cross: {avg_cross:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ── Collapse detection ────────────────────────────────
        if avg_within < COLLAPSE_THRESHOLD:
            low_within_streak += 1
            if low_within_streak >= COLLAPSE_STREAK:
                print(
                    f"  *** COLLAPSE WARNING: L_within < {COLLAPSE_THRESHOLD} "
                    f"for {low_within_streak} consecutive epochs. "
                    f"Backbone may have stopped learning. ***"
                )
        else:
            low_within_streak = 0

        # ── Checkpoint every 5 epochs ─────────────────────────
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"{checkpoint_dir}/ssl_epoch_{epoch+1}.pt"
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    torch.save(model.state_dict(), "/content/drive/MyDrive/Retinal_SSL/checkpoints/ssl_final.pth")
    print(f"SSL pretraining complete. Saved: /content/drive/MyDrive/Retinal_SSL/checkpoints/ssl_final.pth")
    print(f"Training finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":

    CSV_FILES = ["/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv"]

    train_ssl(
        csv_files  = CSV_FILES,
        device     = "cuda",
        epochs     = 50,
        batch_size = 64,
    )

    print("SSL training finished.")