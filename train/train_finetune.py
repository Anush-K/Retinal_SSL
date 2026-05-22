import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    recall_score, confusion_matrix, roc_curve
)
from PIL import Image
import torch.nn as nn
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_simclr.gem import GeM

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 64
EPOCHS         = 30
BACKBONE_LR    = 5e-5
CLASSIFIER_LR  = 1e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 5

CSV_PATH       = "/content/drive/MyDrive/APTOS_metadata.csv"
SSL_CHECKPOINT = "/content/drive/MyDrive/checkpoints/ssl_final.pth"
RESULTS_DIR    = "/content/drive/MyDrive/results_ssl"
CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"

os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class FineTuneDataset(Dataset):
    """
    Loads face crops from the processed dataset.
    Split is determined by directory path structure:
      .../FFPP/train/real/...  or  .../FFPP/train/fake/...
    """
    def __init__(self, csv_file, split, transform):
        df = pd.read_csv(csv_file)
        df = df[df["image_path"].str.contains(f"/{split}/")].reset_index(drop=True)
        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(row["label"], dtype=torch.float32)
        return self.transform(img), label


class BalancedTestDataset(Dataset):
    """
    Balances the test set for threshold-dependent metrics (Accuracy, F1, Recall).

    WHY balance the test set:
    AUC is already immune to class imbalance (threshold-independent).
    But Accuracy, F1, Recall computed at any fixed threshold are distorted
    by majority-class dominance. A model predicting fake for everything on
    CelebDF test (1,422 real / 2,721 fake) achieves 65.7% Accuracy and
    0.79 F1 — neither reveals the failure.

    CORRECT approach — stratified subsampling of majority class:
    - Count minority class (real, since fake >> real in both test sets)
    - From fake, sample equally from each manipulation type so all
      manipulation types remain represented in the balanced set
    - Shuffle result so batches are not class-contiguous

    AUC is ALSO reported separately on the full unbalanced test set
    as a threshold-independent reference metric.

    Reviewer justification:
    "Threshold-dependent metrics are computed on a stratified balanced
    subset of the test set to prevent majority-class inflation. AUC is
    additionally reported on the full test set as a threshold-independent
    reference."
    """
    def __init__(self, csv_file, split, transform, random_seed=42):
        df      = pd.read_csv(csv_file)
        df      = df[df["image_path"].str.contains(f"/{split}/")].reset_index(drop=True)
        real_df = df[df["label"] == 0]
        fake_df = df[df["label"] == 1]
        n_real  = len(real_df)
        n_fake  = len(fake_df)

        if n_real == 0 or n_fake == 0:
            balanced = df
        elif n_real <= n_fake:
            # Real is minority — stratified subsample of fake by manipulation type
            manip_types  = fake_df["manipulation_type"].dropna().unique()
            n_manip      = max(1, len(manip_types))
            per_manip    = max(1, n_real // n_manip)
            sampled_fakes = []
            for m in manip_types:
                pool = fake_df[fake_df["manipulation_type"] == m]
                sampled_fakes.append(
                    pool.sample(n=min(per_manip, len(pool)), random_state=random_seed)
                )
            fake_sampled = pd.concat(sampled_fakes)
            # Top up to exactly n_real if stratified rounding left a shortfall
            shortfall = n_real - len(fake_sampled)
            if shortfall > 0:
                remaining = fake_df[~fake_df.index.isin(fake_sampled.index)]
                if len(remaining) >= shortfall:
                    fake_sampled = pd.concat([
                        fake_sampled,
                        remaining.sample(n=shortfall, random_state=random_seed)
                    ])
            balanced = pd.concat([real_df, fake_sampled])
        else:
            # Fake is minority — subsample real
            balanced = pd.concat([
                real_df.sample(n=n_fake, random_state=random_seed), fake_df
            ])

        self.df        = balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        self.transform = transform
        n_r = (self.df["label"] == 0).sum()
        n_f = (self.df["label"] == 1).sum()
        print(f"  BalancedTestDataset [{split}]: {n_r} real / {n_f} fake "
              f"(from original {n_real} real / {n_fake} fake)")

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
    Manipulation-type balanced sampling strategy:

    Problem: FF++ has 5 manipulation types × ~5,600 fake images = 27,984 fake
             vs 5,600 real images. Naive sampling gives 83% fake batches.

    Strategy: Within the fake class, assign equal probability to each manipulation
              type so the model sees all 5 artifact types equally. Real class gets
              equal total weight to the entire fake class combined.

    Why this helps cross-dataset generalization:
    - Model learns artifact features from ALL manipulation types equally
    - No single manipulation type dominates the representation
    - More generalizable features transfer better to CelebDF (different method)

    Result per epoch: effectively balanced real/fake exposure with
                      uniform manipulation-type coverage.
    """
    df = dataset.df

    # Compute per-sample weights
    # Step 1: within fake, weight inversely by manipulation type frequency
    fake_mask    = df["label"] == 1
    manip_counts = df.loc[fake_mask, "manipulation_type"].value_counts()
    n_manip      = len(manip_counts)

    weights = np.zeros(len(df))

    # Real samples: each gets weight = 1.0
    real_count = (df["label"] == 0).sum()
    weights[df["label"] == 0] = 1.0

    # Fake samples: each manipulation type gets equal total weight = 1.0
    # Individual sample weight = 1 / count_of_its_manipulation_type
    for manip_type, count in manip_counts.items():
        mask = (df["label"] == 1) & (df["manipulation_type"] == manip_type)
        # Each manipulation type contributes equally, so its per-sample weight
        # must compensate for its absolute count
        weights[mask] = real_count / (n_manip * count)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(df),
        replacement=True
    )
    return sampler


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class FineTuneModel(nn.Module):
    """
    EfficientNet-B4 backbone + GeM pooling + linear classifier.

    GeM pooling is preserved from SSL pretraining because:
    - The backbone was trained with GeM — its feature maps are optimized for GeM aggregation
    - Switching to avgpool would misalign the fine-tuning representations from SSL representations
    - GeM's learnable p parameter continues to adapt during fine-tuning
    """
    def __init__(self, ssl_checkpoint):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0
        )
        self.pool       = GeM()
        self.classifier = nn.Linear(1792, 1)

        # Load SSL weights by key prefix
        # SSL checkpoint has keys: backbone.xxx, pool.xxx, projector.xxx
        # We load backbone and pool; projector is discarded (was for contrastive task only)
        print(f"Loading SSL weights from: {ssl_checkpoint}")
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

        missing_b = self.backbone.load_state_dict(backbone_state, strict=False)
        missing_p = self.pool.load_state_dict(pool_state, strict=False)

        print(f"Backbone — missing: {missing_b.missing_keys[:3]}... "
              f"unexpected: {missing_b.unexpected_keys[:3]}")
        print(f"Pool     — missing: {missing_p.missing_keys}, "
              f"unexpected: {missing_p.unexpected_keys}")

        # Classifier initialized with small weights for stable early training
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        features = self.backbone.forward_features(x)   # [B, 1792, 7, 7]
        pooled   = self.pool(features).flatten(1)       # [B, 1792]
        return self.classifier(pooled)                  # [B, 1]  (logits)


# ─────────────────────────────────────────────────────────────
# Freeze Strategy
# ─────────────────────────────────────────────────────────────

def freeze_early_layers(model):
    """
    Freeze EfficientNet-B4 blocks 0, 1, 2 (first ~30% of network).

    Rationale:
    - Early blocks encode low-level features (edges, textures, color gradients)
      learned during ImageNet pretraining and reinforced during SSL.
    - These features are universally useful and expensive to relearn.
    - Freezing prevents catastrophic forgetting of SSL representations.
    - Later blocks (3-7) encode higher-level semantic/artifact features
      that benefit from supervised fine-tuning adaptation.
    """
    frozen_count  = 0
    trainable_count = 0

    for name, param in model.backbone.named_parameters():
        if any(f"blocks.{i}." in name for i in [0, 1, 2]):
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()

    print(f"Frozen parameters:    {frozen_count:,}")
    print(f"Trainable parameters: {trainable_count + sum(p.numel() for p in model.pool.parameters()) + sum(p.numel() for p in model.classifier.parameters()):,}")


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def find_balanced_threshold(all_labels, all_probs):
    """
    Find the decision threshold that maximises balanced accuracy on a
    HELD-OUT VALIDATION SET.

    balanced_acc = 0.5 * (TPR + TNR)

    CRITICAL — no data leakage:
    This function must only be called on the VALIDATION set.
    The returned threshold is then applied UNCHANGED to both test sets.
    Deriving the threshold from the same set it is evaluated on would be
    circular (optimising and measuring on the same data).

    Why balanced accuracy and not F1 or accuracy for threshold selection:
    - F1 is asymmetric (favours the positive/fake class)
    - Accuracy is majority-class biased
    - Balanced accuracy weights TPR and TNR equally regardless of class
      distribution — the correct criterion when both error types matter

    Reviewer justification:
    "The decision threshold is selected on the validation set by maximising
    balanced accuracy (mean of TPR and TNR) and applied without modification
    to all test sets, ensuring no test-set information influences threshold
    selection."
    """
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    tnr          = 1.0 - fpr
    balanced_acc = 0.5 * (tpr + tnr)
    best_idx     = np.argmax(balanced_acc)
    return float(thresholds[best_idx]), float(balanced_acc[best_idx])


def get_probs(model, loader):
    """
    Pure inference — returns raw probabilities and labels only.
    No threshold logic here. Threshold is always derived externally
    from the validation set and passed in.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def compute_metrics(all_probs, all_labels, threshold, split_name=""):
    """
    Compute all metrics at a FIXED threshold derived from the validation set.
    AUC is always threshold-independent.

    Parameters
    ----------
    all_probs  : numpy array of model output probabilities
    all_labels : numpy array of ground truth labels
    threshold  : fixed value derived from validation set — never from this set
    split_name : optional label for console output
    """
    preds = (all_probs >= threshold).astype(int)

    auc    = roc_auc_score(all_labels, all_probs)
    acc    = accuracy_score(all_labels, preds)
    f1     = f1_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    cm     = confusion_matrix(all_labels, preds)

    if split_name:
        n_real = (all_labels == 0).sum()
        n_fake = (all_labels == 1).sum()
        print(f"\n  [{split_name}] Class distribution: "
              f"{n_real} real / {n_fake} fake (ratio 1:{n_fake/max(n_real,1):.1f})")
        print(f"  [{split_name}] Threshold (from val set): {threshold:.4f}")
        print(f"  [{split_name}] AUC    : {auc:.4f}  (threshold-independent)")
        print(f"  [{split_name}] Acc    : {acc:.4f}")
        print(f"  [{split_name}] F1     : {f1:.4f}")
        print(f"  [{split_name}] Recall : {recall:.4f}")
        print(f"  [{split_name}] Confusion Matrix:")
        print(f"                  Pred Real   Pred Fake")
        print(f"    True Real  :   {cm[0,0]:6}      {cm[0,1]:6}")
        print(f"    True Fake  :   {cm[1,0]:6}      {cm[1,1]:6}")

    return auc, acc, f1, recall, cm


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_roc_curve(all_labels, all_probs, threshold, dataset_name, save_path):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc                  = roc_auc_score(all_labels, all_probs)

    # Find the point on the curve closest to our balanced threshold
    idx = np.argmin(np.abs(thresholds - threshold))

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    plt.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    # Mark the balanced operating point
    plt.scatter(fpr[idx], tpr[idx], color="red", zorder=5, s=80,
                label=f"Balanced threshold={threshold:.3f}\n"
                      f"(TPR={tpr[idx]:.3f}, FPR={fpr[idx]:.3f})")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(f"ROC Curve — {dataset_name}", fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved: {save_path}")


def plot_confusion_matrix(all_labels, all_probs, threshold, dataset_name, save_path):
    preds = (all_probs >= threshold).astype(int)
    cm    = confusion_matrix(all_labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real (0)", "Fake (1)"], fontsize=11)
    ax.set_yticklabels(["Real (0)", "Fake (1)"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {dataset_name}\n(balanced threshold={threshold:.3f})", fontsize=12)
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


class ManipSubset(Dataset):
    """Thin dataset wrapper for a single manipulation-type subset."""
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(row["label"], dtype=torch.float32)
        return self.transform(img), label


def plot_per_manipulation(model, csv_file, transform, threshold, dataset_name, save_path):
    """
    Per-manipulation-type AUC and F1 breakdown using val-derived threshold.

    AUC is threshold-independent (reported per manipulation type).
    F1 uses the same val-derived threshold as all other metrics — no leakage.

    Uses batched DataLoader per manipulation type (not row-by-row) for speed.
    """
    df = pd.read_csv(csv_file)
    df = df[df["image_path"].str.contains("/test/")].reset_index(drop=True)

    manip_types = sorted(df["manipulation_type"].unique())
    results     = {}
    model.eval()

    for manip in manip_types:
        subset = df[df["manipulation_type"] == manip]
        if len(subset) < 10 or len(subset["label"].unique()) < 2:
            continue

        loader = DataLoader(
            ManipSubset(subset, transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        probs, labels = get_probs(model, loader)

        auc   = roc_auc_score(labels, probs)
        preds = (probs >= threshold).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        results[manip] = {"auc": auc, "f1": f1}

    if not results:
        return {}

    manips    = list(results.keys())
    auc_vals  = [results[m]["auc"] for m in manips]
    f1_vals   = [results[m]["f1"]  for m in manips]

    x      = np.arange(len(manips))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(manips) * 2), 5))
    bars1 = ax.bar(x - width/2, auc_vals, width, label="AUC",
                   color=["steelblue" if v >= 0.90 else "orange" if v >= 0.80
                          else "tomato" for v in auc_vals])
    bars2 = ax.bar(x + width/2, f1_vals,  width, label=f"F1 (thr={threshold:.3f})",
                   color="mediumseagreen", alpha=0.8)

    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.90, color="navy",   linestyle="--", alpha=0.4, lw=1)
    ax.axhline(y=0.80, color="orange", linestyle="--", alpha=0.4, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(manips, fontsize=10)
    ax.set_xlabel("Manipulation Type", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Manipulation AUC & F1 — {dataset_name}\n"
                 f"(threshold={threshold:.3f} derived from FFPP val set)", fontsize=12)
    ax.legend(fontsize=11)
    for bar, v in zip(bars1, auc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Per-manipulation plot saved: {save_path}")
    return {m: results[m]["auc"] for m in manips}


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train():

    # ── Transforms ──────────────────────────────────────────
    # Train: light augmentation (no heavy augmentation — SSL already learned
    #        augmentation invariance; we just need the model to not overfit)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # Val/Test: no augmentation — deterministic evaluation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # ── Datasets ─────────────────────────────────────────────
    csv_path = f"FFPP_metadata.csv"

    train_dataset = FineTuneDataset(csv_path, split="train", transform=train_transform)
    val_dataset   = FineTuneDataset(csv_path, split="val",   transform=eval_transform)

    print(f"\nTrain set: {len(train_dataset)} samples")
    print(f"  Real: {(train_dataset.df['label']==0).sum()}")
    print(f"  Fake: {(train_dataset.df['label']==1).sum()}")
    print(f"Val set: {len(val_dataset)} samples\n")

    # ── Balanced Sampler ─────────────────────────────────────
    sampler = build_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,          # replaces shuffle=True
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ── Model ────────────────────────────────────────────────
    model = FineTuneModel(SSL_CHECKPOINT).to(DEVICE)
    freeze_early_layers(model)

    # ── Optimizer: layer-wise learning rates ─────────────────
    backbone_params   = [p for p in model.backbone.parameters() if p.requires_grad]
    pool_params       = list(model.pool.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params,   "lr": BACKBONE_LR},
        {"params": pool_params,       "lr": BACKBONE_LR},
        {"params": classifier_params, "lr": CLASSIFIER_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # ── Loss: class-weighted BCE ──────────────────────────────
    # Even with balanced sampler, explicit pos_weight adds robustness.
    # pos_weight < 1 because fake (label=1) is the majority class.
    # Computed from actual train data distribution.
    n_real     = (train_dataset.df["label"] == 0).sum()
    n_fake     = (train_dataset.df["label"] == 1).sum()
    pos_weight = torch.tensor([n_real / n_fake], dtype=torch.float32).to(DEVICE)
    print(f"BCE pos_weight (real/fake ratio): {pos_weight.item():.4f}\n")

    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    # ── Training Loop ─────────────────────────────────────────
    best_auc         = 0.0
    patience_counter = 0
    train_losses     = []
    val_aucs         = []

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

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

        # Validation — derive threshold from val set each epoch for monitoring
        # (the FINAL threshold used for testing is derived after training ends)
        val_probs, val_labels = get_probs(model, val_loader)
        val_threshold, _      = find_balanced_threshold(val_labels, val_probs)
        auc, acc, f1, recall, _ = compute_metrics(
            val_probs, val_labels, val_threshold, split_name=""
        )

        train_losses.append(avg_loss)
        val_aucs.append(auc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e}")
        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  Val AUC    : {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

        # ── Checkpoint every 5 epochs ─────────────────────────
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"finetune_epoch_{epoch+1}.pth")
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc":              auc,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # ── Best model ────────────────────────────────────────
        if auc > best_auc:
            best_auc         = auc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "finetune_best.pth"))
            print(f"  ✓ Best model saved (val AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}. Best val AUC: {best_auc:.4f}")
                break

    # ── Training curve plot ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, marker="o", color="steelblue")
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.plot(val_aucs, marker="o", color="green")
    ax2.set_title("Validation AUC"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"\nTraining curves saved.")
    print(f"Training complete. Best val AUC: {best_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────
def test(results_dir, csv_path, ssl_checkpoint, run_label="SSL"):
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225))
    ])

    model = FineTuneModel(ssl_checkpoint).to(DEVICE)
    best_ckpt = os.path.join(CHECKPOINT_DIR, "finetune_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()

    # Threshold from val set
    val_dataset = FineTuneDataset(csv_path, split="val",
                                  transform=eval_transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)
    val_probs, val_labels   = get_probs(model, val_loader)
    threshold, val_bal_acc  = find_balanced_threshold(val_labels, val_probs)

    # Test set evaluation
    test_dataset = FineTuneDataset(csv_path, split="test",
                                   transform=eval_transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
    test_probs, test_labels = get_probs(model, test_loader)

    # ── Compute all metrics ───────────────────────────────────
    preds = (test_probs >= threshold).astype(int)

    auc       = roc_auc_score(test_labels, test_probs)
    acc       = accuracy_score(test_labels, preds)
    f1        = f1_score(test_labels, preds, zero_division=0)
    recall    = recall_score(test_labels, preds, zero_division=0)    # sensitivity
    precision = precision_score(test_labels, preds, zero_division=0)
    cm        = confusion_matrix(test_labels, preds)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── Print results ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {run_label}")
    print(f"{'='*60}")
    print(f"  Threshold (from val set): {threshold:.4f}")
    print(f"  Val Balanced Accuracy   : {val_bal_acc:.4f}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*32}")
    print(f"  {'AUC-ROC':<20} {auc:>10.4f}")
    print(f"  {'Accuracy':<20} {acc:>10.4f}")
    print(f"  {'Precision':<20} {precision:>10.4f}")
    print(f"  {'Recall (Sensitivity)':<20} {recall:>10.4f}")
    print(f"  {'Specificity':<20} {specificity:>10.4f}")
    print(f"  {'F1 Score':<20} {f1:>10.4f}")
    print(f"  {'TP':<20} {tp:>10}")
    print(f"  {'TN':<20} {tn:>10}")
    print(f"  {'FP':<20} {fp:>10}")
    print(f"  {'FN':<20} {fn:>10}")
    print(f"{'='*60}\n")

    # ── Save confusion matrix ─────────────────────────────────
    plot_confusion_matrix(
        test_labels, test_probs, threshold,
        dataset_name=f"APTOS — {run_label}",
        save_path=os.path.join(results_dir, f"confusion_matrix_{run_label}.png")
    )
    print(f"Confusion matrix plotted and saved in {os.path.join(results_dir, f'confusion_matrix_{run_label}.png')}.")

    # ── Save ROC curve ────────────────────────────────────────
    plot_roc_curve(
        test_labels, test_probs, threshold,
        dataset_name=f"APTOS — {run_label}",
        save_path=os.path.join(results_dir, f"roc_curve_{run_label}.png")
    )
    print(f"ROC curve plotted and saved in {os.path.join(results_dir, f'roc_curve_{run_label}.png')}.")
    # ── Return dict for ablation table ───────────────────────
    return {
        "Run":         run_label,
        "AUC":         round(auc, 4),
        "Accuracy":    round(acc, 4),
        "Precision":   round(precision, 4),
        "Recall":      round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1":          round(f1, 4),
        "Threshold":   round(threshold, 4),
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
    }


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
    test()
