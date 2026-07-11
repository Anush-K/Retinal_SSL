"""
ssl_dataset.py — Dataset for LSFC self-supervised pretraining

Always generates all 5 views (view_s1, view_s2, view_f_fine, view_f_coarse,
view_f_legacy); train_ssl.py's --mode flag selects which are actually used
in the loss, so a single dataset class serves all three ablation modes
without duplicated data-loading code.
"""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ssl_simclr.augmentations import get_ssl_transforms


class SSLDataset(Dataset):
    def __init__(self, csv_files, image_size: int = 224):
        self.df = pd.concat(
            [pd.read_csv(f) for f in csv_files], ignore_index=True
        )
        self.df = self.df[
            ~self.df["image_path"].str.contains("/test/")
        ].reset_index(drop=True)

        (self.base_aug, self.hard_aug,
         self.highpass_fine_aug, self.highpass_coarse_aug,
         self.highpass_legacy_aug) = get_ssl_transforms(image_size)

        print(f"[SSLDataset] Loaded {len(self.df)} samples "
              f"from {len(csv_files)} CSV(s). Test splits excluded.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            alt_idx = (idx + 1) % len(self.df)
            img = Image.open(self.df.iloc[alt_idx]["image_path"]).convert("RGB")

        view_s1     = self.base_aug(img)
        view_s2     = self.hard_aug(img)
        view_fine   = self.highpass_fine_aug(img)
        view_coarse = self.highpass_coarse_aug(img)
        view_legacy = self.highpass_legacy_aug(img)

        return view_s1, view_s2, view_fine, view_coarse, view_legacy