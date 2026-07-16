"""
check_band_redundancy.py — quick diagnostic (no retraining needed)

Checks whether the fine-band and coarse-band frequency embeddings from the
trained multi_band checkpoint are actually distinct, or redundant (which
the near-identical L_fine/L_coarse curves during training suggest).

If mean cosine similarity is very high (>0.9), the two heads have
converged to near-duplicate representations — evidence that GeM pooling
loses the scale information before the frequency heads can use it.
If similarity is moderate/low, the heads ARE differentiated, and the null
result on downstream metrics has a different explanation (e.g. the
classifier just doesn't need scale-specific frequency info for this task).

Run from Colab:
    !PYTHONPATH=/content/Retinal_SSL python train/check_band_redundancy.py
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from ssl_simclr.ssl_model import SSLModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SSLModel(mode="multi_band").to(device)
model.load_state_dict(torch.load(
    "/content/drive/MyDrive/Retinal_SSL/checkpoints_multi_band/ssl_final.pth",
    map_location=device
))
model.eval()

df = pd.read_csv("/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv")
df = df[~df["image_path"].str.contains("/test/")].reset_index(drop=True)
df = df.sample(n=min(300, len(df)), random_state=42).reset_index(drop=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class SimpleDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        return self.transform(img)

loader = DataLoader(SimpleDataset(df, transform), batch_size=32, num_workers=4)

sims = []
with torch.no_grad():
    for imgs in loader:
        imgs = imgs.to(device)
        pooled = model.encode(imgs)
        z_fine   = F.normalize(model.projector.forward_freq_fine(pooled), dim=1)
        z_coarse = F.normalize(model.projector.forward_freq_coarse(pooled), dim=1)
        cos_sim = (z_fine * z_coarse).sum(dim=1)  # per-sample cosine sim
        sims.extend(cos_sim.cpu().tolist())

sims = torch.tensor(sims)
print(f"N samples: {len(sims)}")
print(f"Mean cosine similarity (fine vs coarse embedding): {sims.mean():.4f}")
print(f"Std : {sims.std():.4f}  Min: {sims.min():.4f}  Max: {sims.max():.4f}")
print()
if sims.mean() > 0.9:
    print("HIGH similarity — the two frequency heads have converged to")
    print("near-redundant representations. This supports the hypothesis")
    print("that global pooling loses scale information before the")
    print("frequency branches can differentiate on it.")
elif sims.mean() > 0.6:
    print("MODERATE similarity — some differentiation, but substantial overlap.")
else:
    print("LOW similarity — the two heads ARE learning distinct representations.")
    print("The null downstream result likely has a different explanation.")