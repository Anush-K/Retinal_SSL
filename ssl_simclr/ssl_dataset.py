"""
ssl_dataset.py — Dataset for DBFC self-supervised pretraining

Changes from original:
1. Generates THREE views per image instead of two:
   - view_s1 : spatial augmentation (base_aug)
   - view_s2 : second independent spatial augmentation (base_aug2)
   - view_f  : frequency-enhanced view (highpass_aug), generated with
               probability p_freq (default 1.0 during pretraining — always
               generate a frequency view, but the loss weights it at 0.3
               so its influence is controlled there, not here)

2. The (view_s1, view_s2) pair drives the within-domain contrastive loss.
   The (view_s1, view_f) pair drives the cross-domain alignment loss.
   This is handled in ssl_model.py / contrastive_loss.py.

3. Test-split exclusion preserved from original.
"""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ssl_simclr.augmentations import get_ssl_transforms


class SSLDataset(Dataset):
    """
    Loads unlabeled face crops for DBFC contrastive pretraining.

    Args:
        csv_files : list of metadata CSV paths (FFPP_metadata.csv, CelebDF_metadata.csv)
        image_size: spatial resolution (default 224)
        p_freq    : probability of generating a frequency view for each sample.
                    Always 1.0 here — the cross-domain loss weight (λ=0.3) in
                    contrastive_loss.py controls its influence instead.
    """

    def __init__(self, csv_files, image_size: int = 224, p_freq: float = 1.0):
        self.df = pd.concat(
            [pd.read_csv(f) for f in csv_files],
            ignore_index=True
        )
        # Strictly exclude test splits — no data leakage into pretraining
        self.df = self.df[
            ~self.df["image_path"].str.contains("/test/")
        ].reset_index(drop=True)

        self.p_freq = p_freq
        self.base_aug, self.hard_aug, self.highpass_aug = get_ssl_transforms(image_size)

        print(f"[SSLDataset] Loaded {len(self.df)} samples "
              f"from {len(csv_files)} CSV(s). Test splits excluded.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # If an image is corrupt, return a neighbouring sample
            # This avoids crashing a long pretraining run over one bad file
            alt_idx = (idx + 1) % len(self.df)
            img = Image.open(self.df.iloc[alt_idx]["image_path"]).convert("RGB")

        view_s1 = self.base_aug(img)   # spatial view 1
        view_s2 = self.hard_aug(img)  # spatial view 2 (independent random state)

        # Frequency view — always generated (influence controlled by λ in loss)
        view_f = self.highpass_aug(img)

        return view_s1, view_s2, view_f