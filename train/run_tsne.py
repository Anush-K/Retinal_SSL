"""
run_tsne.py — t-SNE visualization of SSL representations for APTOS
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from ssl_simclr.ssl_model import SSLModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load SSL model ─────────────────────────────────────────────
model = SSLModel().to(device)
model.load_state_dict(
    torch.load("ssl_final.pth", map_location=device), strict=False
)
model.eval()

# ── Load metadata ──────────────────────────────────────────────
df = pd.read_csv("/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv")

# Exclude test split — visualize train+val representations only
df = df[~df["image_path"].str.contains("/test/")].reset_index(drop=True)

# Balanced sample — up to 500 per class
dfs = []
for lbl in [0, 1]:
    subset = df[df["label"] == lbl]
    dfs.append(subset.sample(n=min(500, len(subset)), random_state=42))

df_tsne = pd.concat(dfs).reset_index(drop=True)
print(f"t-SNE samples: {len(df_tsne)} "
      f"({(df_tsne.label==0).sum()} NORMAL, "
      f"{(df_tsne.label==1).sum()} ABNORMAL)")

# ── Transform ──────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

class EvalDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        return self.transform(img), row["label"]

eval_dataset = EvalDataset(df_tsne, transform)
eval_loader  = DataLoader(eval_dataset, batch_size=128,
                          num_workers=4, pin_memory=True)

# ── Extract features ───────────────────────────────────────────
features, labels = [], []

for imgs, lbls in tqdm(eval_loader, desc="Extracting features"):
    imgs = imgs.to(device)
    with torch.no_grad():
        feat_map = model.backbone.forward_features(imgs)
        pooled   = model.pool(feat_map).flatten(1)
    features.append(pooled.cpu().numpy())
    labels.append(lbls.numpy())

features = np.concatenate(features)
labels   = np.concatenate(labels)
print(f"Feature shape: {features.shape}")

# ── Run t-SNE ──────────────────────────────────────────────────
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
features_2d = tsne.fit_transform(features)
print("t-SNE completed.")

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

colors = {0: "#2196F3", 1: "#F44336"}
names  = {0: "NORMAL", 1: "ABNORMAL"}

for lbl in [0, 1]:
    mask = labels == lbl
    ax.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        c=colors[lbl],
        label=names[lbl],
        alpha=0.6,
        s=15,
        edgecolors="none"
    )

ax.set_title("t-SNE: SSL Representations — APTOS\n(NORMAL vs ABNORMAL)",
             fontsize=14)
ax.legend(fontsize=12)
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
plt.tight_layout()

os.makedirs("/content/drive/MyDrive/Retinal_SSL/results_ssl_dbfc", exist_ok=True)
save_path = "/content/drive/MyDrive/Retinal_SSL/results_ssl_dbfc/tsne_ssl_aptos.png"
plt.savefig(save_path, dpi=150)
plt.show()
print(f"Saved: {save_path}")