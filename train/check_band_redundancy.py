"""
check_band_redundancy.py — quick diagnostic (no retraining needed)

Checks whether the fine-band and coarse-band frequency embeddings from a
trained multi_band or multi_band_sp checkpoint are distinct or redundant.

Usage (from Colab):
    !PYTHONPATH=/content/Retinal_SSL python train/check_band_redundancy.py --mode multi_band_sp
    !PYTHONPATH=/content/Retinal_SSL python train/check_band_redundancy.py --mode multi_band
"""

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

from ssl_simclr.ssl_model import SSLModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True,
                    choices=["multi_band", "multi_band_sp"])
    p.add_argument("--csv_path", type=str,
                    default="/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv")
    return p.parse_args()


class SimpleDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        return self.transform(img)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = f"/content/drive/MyDrive/Retinal_SSL/checkpoints_{args.mode}/ssl_final.pth"

    model = SSLModel(mode=args.mode).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    df = pd.read_csv(args.csv_path)
    df = df[~df["image_path"].str.contains("/test/")].reset_index(drop=True)
    df = df.sample(n=min(300, len(df)), random_state=42).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    loader = DataLoader(SimpleDataset(df, transform), batch_size=32, num_workers=4)

    sims = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            # forward() already returns L2-normalized 128-d embeddings —
            # this is exactly what the contrastive loss operates on, so
            # comparing here is the most direct test of redundancy.
            z_fine   = model(imgs, domain="freq_fine")
            z_coarse = model(imgs, domain="freq_coarse")
            cos_sim = (z_fine * z_coarse).sum(dim=1)  # already normalized -> dot = cosine
            sims.extend(cos_sim.cpu().tolist())

    sims = torch.tensor(sims)
    print(f"\nMode: {args.mode}")
    print(f"N samples: {len(sims)}")
    print(f"Mean cosine similarity (fine vs coarse embedding): {sims.mean():.4f}")
    print(f"Std : {sims.std():.4f}  Min: {sims.min():.4f}  Max: {sims.max():.4f}")
    print()
    if sims.mean() > 0.9:
        print("HIGH similarity — heads are still near-redundant.")
    elif sims.mean() > 0.6:
        print("MODERATE similarity — partial differentiation.")
    else:
        print("LOW similarity — heads have learned distinct representations.")


if __name__ == "__main__":
    main()