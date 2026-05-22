"""
train_finetune.py — Supervised fine-tuning with DBFC SSL pretrained weights (APTOS)

Pipeline:
  1. Load SSL-pretrained backbone + GeM from ssl_final.pth
  2. Freeze blocks 0-2 (low-level features preserved from SSL)
  3. Fine-tune on APTOS train split with label-smoothed BCE loss
  4. Derive balanced threshold from validation set
  5. Evaluate on test set — report AUC, Accuracy, Precision, Recall,
     Specificity, F1, TP, TN, FP, FN, Confusion Matrix, ROC curve,
     Training curves
  6. Save evaluation_summary.csv for ablation table

Usage (from Colab):
    python train/train_finetune.py \
        --run_label DBFC_SSL \
        --results_dir /content/drive/MyDrive/Retinal_SSL/results_ssl_dbfc \
        --ssl_checkpoint /content/drive/MyDrive/Retinal_SSL/checkpoints/ssl_final.pth \
        --checkpoint_dir /content/drive/MyDrive/Retinal_SSL/checkpoints_ssl
"""

import os
import sys
import argparse
import datetime
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    recall_score, precision_score, confusion_matrix, roc_curve
)
from PIL import Image
import torch.nn as nn
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup so ssl_simclr imports resolve when called from train/ ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_simclr.gem import GeM

# ─────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune with SSL weights on APTOS"
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="DBFC_SSL",
        help="Label for this run (used in filenames and ablation table)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/content/drive/MyDrive/Retinal_SSL/results_ssl_dbfc",
        help="Directory to save all result plots and CSVs"
    )
    parser.add_argument(
        "--ssl_checkpoint",
        type=str,
        default="/content/drive/MyDrive/Retinal_SSL/checkpoints/ssl_final.pth",
        help="Path to ssl_final.pth from pretraining"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/content/drive/MyDrive/Retinal_SSL/checkpoints_ssl",
        help="Directory to save fine-tuning checkpoints"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv",
        help="Path to APTOS metadata CSV"
    )
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--backbone_lr",  type=float, default=5e-5)
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=5)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class FineTuneDataset(Dataset):
    """
    Loads APTOS images from the metadata CSV.
    Split membership is determined by the path string containing
    /train/, /val/, or /test/ — set by prepare_dataset.py.

    label: 0 = NORMAL, 1 = ABNORMAL
    """
    def __init__(self, csv_file, split, transform):
        df = pd.read_csv(csv_file)
        df = df[df["image_path"].str.contains(f"/{split}/")].reset_index(drop=True)
        self.df        = df
        self.transform = transform
        print(f"  [{split}] {len(df)} samples — "
              f"{(df['label']==0).sum()} NORMAL, "
              f"{(df['label']==1).sum()} ABNORMAL")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(row["label"], dtype=torch.float32)
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────
# Balanced Sampler
# ─────────────────────────────────────────────────────────────

def build_balanced_sampler(dataset):
    """
    Simple class-balanced sampling for APTOS.

    APTOS has only two manipulation_type values: 'normal' and 'abnormal'.
    We give each class equal total weight regardless of raw count.

    With ~1263 NORMAL and ~1300 ABNORMAL in train (near-balanced),
    this sampler has a small effect but keeps the framework consistent
    and prevents any imbalance from creeping in after the 70/15/15 split.
    """
    df         = dataset.df
    n_normal   = (df["label"] == 0).sum()
    n_abnormal = (df["label"] == 1).sum()
    n_total    = len(df)

    weights = np.zeros(n_total)
    # Each class gets weight proportional to the inverse of its count
    weights[df["label"] == 0] = 1.0 / n_normal
    weights[df["label"] == 1] = 1.0 / n_abnormal

    # Normalise so average weight = 1 (keeps effective batch size stable)
    weights = weights / weights.sum() * n_total

    sampler = WeightedRandomSampler(
        weights    = torch.DoubleTensor(weights),
        num_samples= n_total,
        replacement= True
    )
    print(f"  Balanced sampler: {n_normal} NORMAL, {n_abnormal} ABNORMAL")
    return sampler


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class FineTuneModel(nn.Module):
    """
    EfficientNet-B4 + GeM pooling + linear classifier.

    Loaded from SSL checkpoint: backbone.* and pool.* keys are extracted.
    projector.* keys are silently ignored (discarded after SSL pretraining).
    Classifier: Linear(1792 → 1), randomly initialised.

    Output: raw logit (scalar). Apply sigmoid for probability.
    """
    def __init__(self, ssl_checkpoint):
        super().__init__()

        self.backbone   = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0
        )
        self.pool       = GeM()
        self.classifier = nn.Linear(1792, 1)

        print(f"\nLoading SSL weights from: {ssl_checkpoint}")
        ssl_state = torch.load(ssl_checkpoint, map_location="cpu")

        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in ssl_state.items()
            if k.startswith("backbone.")
        }
        pool_state = {
            k.replace("pool.", ""): v
            for k, v in ssl_state.items()
            if k.startswith("pool.")
        }

        miss_b = self.backbone.load_state_dict(backbone_state, strict=False)
        miss_p = self.pool.load_state_dict(pool_state, strict=False)

        print(f"  Backbone missing : {miss_b.missing_keys[:2]}"
              f"  unexpected: {miss_b.unexpected_keys[:2]}")
        print(f"  Pool missing     : {miss_p.missing_keys}"
              f"  unexpected: {miss_p.unexpected_keys}")

        # Small weight init → stable early training
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        features = self.backbone.forward_features(x)  # [B, 1792, 7, 7]
        pooled   = self.pool(features).flatten(1)      # [B, 1792]
        return self.classifier(pooled)                 # [B, 1]  logits


# ─────────────────────────────────────────────────────────────
# Freeze Strategy
# ─────────────────────────────────────────────────────────────

def freeze_early_layers(model):
    """
    Freeze EfficientNet-B4 blocks 0, 1, 2.

    These early blocks encode low-level features (edges, textures)
    learned during ImageNet pretraining and reinforced during SSL.
    Freezing prevents the supervised loss (trained only on APTOS)
    from overwriting these general representations.
    Blocks 3-7 adapt to retinal image features.
    """
    frozen_count    = 0
    trainable_count = 0

    for name, param in model.backbone.named_parameters():
        if any(f"blocks.{i}." in name for i in [0, 1, 2]):
            param.requires_grad = False
            frozen_count       += param.numel()
        else:
            trainable_count    += param.numel()

    total_trainable = (trainable_count
                       + sum(p.numel() for p in model.pool.parameters())
                       + sum(p.numel() for p in model.classifier.parameters()))

    print(f"  Frozen (blocks 0-2)  : {frozen_count:,} params")
    print(f"  Trainable            : {total_trainable:,} params")


# ─────────────────────────────────────────────────────────────
# Label Smoothing Loss
# ─────────────────────────────────────────────────────────────

class SmoothedBCE(nn.Module):
    """
    BCE with label smoothing (ε = 0.05).

    Hard targets (0, 1) are replaced with (0.05, 0.95).
    This prevents overconfidence on APTOS-specific patterns,
    keeping the decision boundary more flexible.
    """
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce       = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smooth = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return self.bce(logits, smooth)


# ─────────────────────────────────────────────────────────────
# Evaluation Utilities
# ─────────────────────────────────────────────────────────────

def get_probs(model, loader, device):
    """Run inference; return (probs, labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def find_balanced_threshold(labels, probs):
    """
    Select threshold that maximises balanced accuracy on the validation set.
    balanced_acc = 0.5 * (TPR + TNR)

    This function is called ONLY on the validation set.
    The returned threshold is applied unchanged to the test set.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    tnr          = 1.0 - fpr
    balanced_acc = 0.5 * (tpr + tnr)
    best_idx     = np.argmax(balanced_acc)
    return float(thresholds[best_idx]), float(balanced_acc[best_idx])


def compute_all_metrics(probs, labels, threshold):
    """
    Compute all classification metrics at a fixed threshold.
    Returns a dict with all values.
    """
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    auc         = roc_auc_score(labels, probs)
    acc         = accuracy_score(labels, preds)
    precision   = precision_score(labels, preds, zero_division=0)
    recall      = recall_score(labels, preds, zero_division=0)      # sensitivity
    f1          = f1_score(labels, preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc":         auc,
        "accuracy":    acc,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "f1":          f1,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "threshold":   threshold,
        "cm":          cm,
    }


def print_metrics(metrics, run_label, val_bal_acc):
    """Print a clean formatted metrics table."""
    print(f"\n{'='*62}")
    print(f"  RESULTS — {run_label}")
    print(f"{'='*62}")
    print(f"  Threshold (from val set) : {metrics['threshold']:.4f}")
    print(f"  Val Balanced Accuracy    : {val_bal_acc:.4f}")
    print(f"  {'─'*40}")
    print(f"  {'Metric':<26} {'Value':>10}")
    print(f"  {'─'*40}")
    print(f"  {'AUC-ROC':<26} {metrics['auc']:>10.4f}")
    print(f"  {'Accuracy':<26} {metrics['accuracy']:>10.4f}")
    print(f"  {'Precision':<26} {metrics['precision']:>10.4f}")
    print(f"  {'Recall  (Sensitivity)':<26} {metrics['recall']:>10.4f}")
    print(f"  {'Specificity':<26} {metrics['specificity']:>10.4f}")
    print(f"  {'F1 Score':<26} {metrics['f1']:>10.4f}")
    print(f"  {'─'*40}")
    print(f"  {'TP (ABNORMAL correct)':<26} {metrics['tp']:>10}")
    print(f"  {'TN (NORMAL   correct)':<26} {metrics['tn']:>10}")
    print(f"  {'FP (NORMAL → ABNORMAL)':<26} {metrics['fp']:>10}")
    print(f"  {'FN (ABNORMAL → NORMAL)':<26} {metrics['fn']:>10}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_roc(labels, probs, threshold, run_label, save_path):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    # Find operating point closest to the chosen threshold
    idx = np.argmin(np.abs(thresholds - threshold))

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2.5,
             label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    plt.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    plt.scatter(fpr[idx], tpr[idx], color="crimson", zorder=5, s=100,
                label=f"Threshold = {threshold:.3f}\n"
                      f"(Sensitivity={tpr[idx]:.3f}, "
                      f"Specificity={1-fpr[idx]:.3f})")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("1 − Specificity  (FPR)", fontsize=13)
    plt.ylabel("Sensitivity  (TPR)", fontsize=13)
    plt.title(f"ROC Curve — APTOS\n{run_label}", fontsize=13)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved    : {save_path}")


def plot_confusion(cm, run_label, save_path):
    labels = ["NORMAL\n(0)", "ABNORMAL\n(1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="gray",
        ax=ax, annot_kws={"size": 14}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — APTOS\n{run_label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_training_curves(train_losses, val_aucs, run_label, save_path):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, marker="o", color="steelblue",
             markersize=4, lw=2, label="Train Loss")
    ax1.set_title(f"Training Loss — {run_label}", fontsize=12)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.grid(alpha=0.3); ax1.legend()

    # AUC curve
    ax2.plot(epochs, val_aucs, marker="o", color="seagreen",
             markersize=4, lw=2, label="Val AUC")
    ax2.axhline(y=max(val_aucs), color="gray", linestyle="--",
                alpha=0.6, label=f"Best = {max(val_aucs):.4f}")
    ax2.set_title(f"Validation AUC — {run_label}", fontsize=12)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_ylim(
        max(0.4, min(val_aucs) - 0.05),
        min(1.02, max(val_aucs) + 0.05)
    )
    ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved : {save_path}")


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*62}")
    print(f"  TRAINING — {args.run_label}")
    print(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device  : {device}")
    print(f"{'='*62}")

    os.makedirs(args.results_dir,    exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────
    # Train: mild augmentation — SSL already learned augmentation invariance
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),   # retinal images can be flipped
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    # Val/Test: deterministic — no augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    # ── Datasets ──────────────────────────────────────────────
    print("\nDataset splits:")
    train_dataset = FineTuneDataset(args.csv_path, "train", train_transform)
    val_dataset   = FineTuneDataset(args.csv_path, "val",   eval_transform)

    sampler = build_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size        = args.batch_size,
        sampler           = sampler,
        num_workers       = 4,
        pin_memory        = True,
        persistent_workers= True,
        drop_last         = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────
    model = FineTuneModel(args.ssl_checkpoint).to(device)
    freeze_early_layers(model)

    # ── Optimizer ─────────────────────────────────────────────
    backbone_params   = [p for p in model.backbone.parameters()
                         if p.requires_grad]
    pool_params       = list(model.pool.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params,   "lr": args.backbone_lr},
        {"params": pool_params,       "lr": args.backbone_lr},
        {"params": classifier_params, "lr": args.classifier_lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    criterion = SmoothedBCE(smoothing=0.05)
    scaler    = torch.cuda.amp.GradScaler()

    # ── Training Loop ─────────────────────────────────────────
    best_auc         = 0.0
    patience_counter = 0
    train_losses     = []
    val_aucs         = []

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(patience={args.patience})...\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs}"
        ):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(imgs).squeeze(1)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_probs, val_labels   = get_probs(model, val_loader, device)
        val_threshold, val_bacc = find_balanced_threshold(val_labels, val_probs)
        val_metrics = compute_all_metrics(val_probs, val_labels, val_threshold)

        train_losses.append(avg_loss)
        val_aucs.append(val_metrics["auc"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e}")
        print(f"  Train Loss   : {avg_loss:.4f}")
        print(f"  Val AUC      : {val_metrics['auc']:.4f}  "
              f"Acc: {val_metrics['accuracy']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}  "
              f"Recall: {val_metrics['recall']:.4f}  "
              f"Spec: {val_metrics['specificity']:.4f}")

        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(
                args.checkpoint_dir, f"finetune_epoch_{epoch+1}.pth"
            )
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc":              val_metrics["auc"],
            }, ckpt)
            print(f"  Checkpoint saved: {ckpt}")

        # Best model
        if val_metrics["auc"] > best_auc:
            best_auc         = val_metrics["auc"]
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_dir, "finetune_best.pth")
            )
            print(f"  ✓ Best model saved (val AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}. "
                      f"Best val AUC: {best_auc:.4f}")
                break

    # Training curves
    plot_training_curves(
        train_losses, val_aucs,
        run_label = args.run_label,
        save_path = os.path.join(
            args.results_dir,
            f"training_curves_{args.run_label}.png"
        )
    )
    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    print(f"Finished : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    # ── Load best checkpoint ───────────────────────────────────
    model     = FineTuneModel(args.ssl_checkpoint).to(device)
    best_ckpt = os.path.join(args.checkpoint_dir, "finetune_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    print(f"\nLoaded best checkpoint: {best_ckpt}")

    # ── Step 1: Derive threshold from validation set ───────────
    print(f"\n{'='*62}")
    print("STEP 1: Deriving threshold from validation set")
    print("(Threshold is fixed here and applied unchanged to test set)")
    print(f"{'='*62}")

    val_dataset = FineTuneDataset(args.csv_path, "val", eval_transform)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    val_probs, val_labels     = get_probs(model, val_loader, device)
    threshold, val_bal_acc    = find_balanced_threshold(val_labels, val_probs)

    print(f"  Val set       : "
          f"{(val_labels==0).sum()} NORMAL, "
          f"{(val_labels==1).sum()} ABNORMAL")
    print(f"  Threshold     : {threshold:.4f}")
    print(f"  Val Bal. Acc. : {val_bal_acc:.4f}")

    # ── Step 2: Evaluate on test set ──────────────────────────
    print(f"\n{'='*62}")
    print("STEP 2: Test set evaluation (APTOS)")
    print(f"{'='*62}")

    test_dataset = FineTuneDataset(args.csv_path, "test", eval_transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    test_probs, test_labels = get_probs(model, test_loader, device)
    metrics = compute_all_metrics(test_probs, test_labels, threshold)

    print_metrics(metrics, args.run_label, val_bal_acc)

    # ── Plots ──────────────────────────────────────────────────
    plot_roc(
        test_labels, test_probs, threshold,
        run_label = args.run_label,
        save_path = os.path.join(
            args.results_dir,
            f"roc_curve_{args.run_label}.png"
        )
    )
    plot_confusion(
        metrics["cm"],
        run_label = args.run_label,
        save_path = os.path.join(
            args.results_dir,
            f"confusion_matrix_{args.run_label}.png"
        )
    )

    # ── Save evaluation_summary.csv ───────────────────────────
    summary = {
        "Run":         args.run_label,
        "AUC":         round(metrics["auc"],         4),
        "Accuracy":    round(metrics["accuracy"],     4),
        "Precision":   round(metrics["precision"],    4),
        "Recall":      round(metrics["recall"],       4),
        "Specificity": round(metrics["specificity"],  4),
        "F1":          round(metrics["f1"],           4),
        "Threshold":   round(threshold,               4),
        "TP":          metrics["tp"],
        "TN":          metrics["tn"],
        "FP":          metrics["fp"],
        "FN":          metrics["fn"],
    }
    summary_path = os.path.join(args.results_dir, "evaluation_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Summary CSV saved  : {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir,    exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
    test(args)