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

device = "cuda"

# -------- Load Trained SSL Model --------
model = SSLModel().to(device)
model.load_state_dict(torch.load("ssl_final.pth", map_location=device))
model.eval()

# -------- Load Metadata --------
df = pd.concat([
    pd.read_csv("FFPP_metadata.csv"),
    pd.read_csv("CelebDF_metadata.csv")
])

# Remove test split
df = df[~df["image_path"].str.contains("/test/")].reset_index(drop=True)

# Balanced sample — 500 per category
dfs = []
for ds in ["FFPP", "CelebDF"]:
    for lbl in [0, 1]:
        subset = df[(df["dataset"] == ds) & (df["label"] == lbl)]
        dfs.append(subset.sample(n=min(500, len(subset)), random_state=42))

df_tsne = pd.concat(dfs).reset_index(drop=True)
print("t-SNE samples:", len(df_tsne))

# -------- Transform --------
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
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        return self.transform(img), row["label"], row["dataset"]

eval_dataset = EvalDataset(df_tsne, transform)
eval_loader  = DataLoader(eval_dataset, batch_size=128,
                          num_workers=4, pin_memory=True)

features, labels, datasets = [], [], []

for imgs, lbls, dsets in tqdm(eval_loader):
    imgs = imgs.to(device)
    with torch.no_grad():
        feat_map = model.backbone.forward_features(imgs)
        pooled   = model.pool(feat_map).flatten(1)
    features.append(pooled.cpu().numpy())
    labels.append(lbls.numpy())
    datasets.extend(dsets)

features = np.concatenate(features)
labels   = np.concatenate(labels)
datasets = np.array(datasets)

print("Feature shape:", features.shape)

# -------- Run t-SNE --------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

features_2d = tsne.fit_transform(features)
print("t-SNE completed.")

plt.figure(figsize=(14, 6))

# -------- Plot 1: Real vs Fake --------
plt.subplot(1, 2, 1)
plt.scatter(features_2d[labels==0, 0], features_2d[labels==0, 1],
            alpha=0.5, s=10, label="Real")
plt.scatter(features_2d[labels==1, 0], features_2d[labels==1, 1],
            alpha=0.5, s=10, label="Fake")
plt.title("t-SNE: Real vs Fake")
plt.legend()

# -------- Plot 2: Dataset Coloring --------
plt.subplot(1, 2, 2)
plt.scatter(features_2d[datasets=="FFPP", 0], features_2d[datasets=="FFPP", 1],
            alpha=0.5, s=10, label="FFPP")
plt.scatter(features_2d[datasets=="CelebDF", 0], features_2d[datasets=="CelebDF", 1],
            alpha=0.5, s=10, label="CelebDF")
plt.title("t-SNE: Dataset Coloring")
plt.legend()

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/tsne_ssl_df.png", dpi=150)

print("Saved to results/tsne_ssl_df.png")