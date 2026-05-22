"""
prepare_dataset.py
Reads raw APTOS folder from Drive, creates train/val/test split,
and generates APTOS_metadata.csv.

Run once in Colab before any training.
Usage: python prepare_dataset.py
"""

import os
import random
import shutil
import pandas as pd
from tqdm import tqdm

random.seed(42)

# ── Paths ─────────────────────────────────────────────────────
RAW_BASE       = "/content/drive/MyDrive/APTOS"
PROCESSED_BASE = "/content/drive/MyDrive/Retinal_SSL/APTOS_processed"
CSV_SAVE_PATH  = "/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv"

# ── Collect all images from raw folder ────────────────────────
def collect_images(raw_base):
    """
    Pools all images from the raw APTOS folder regardless of
    original train/test split. Returns dict {class: [paths]}.
    """
    class_map = {"NORMAL": 0, "ABNORMAL": 1}
    all_images = {"NORMAL": [], "ABNORMAL": []}

    for split_folder in tqdm(["train", "test"], desc="Scanning splits"):
        for class_name in ["NORMAL", "ABNORMAL"]:
            folder = os.path.join(raw_base, split_folder, class_name)
            if not os.path.exists(folder):
                print(f"  [SKIP] Not found: {folder}")
                continue
            imgs = [
                os.path.join(folder, f)
                for f in sorted(os.listdir(folder))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            all_images[class_name].extend(imgs)
            print(f"  [FOUND] {folder}: {len(imgs)} images")

    return all_images

# ── Split and copy ─────────────────────────────────────────────
def create_split(all_images, processed_base):
    """
    70% train / 15% val / 15% test, per class.
    Copies images into processed_base/split/class/ structure.
    Returns list of metadata entries.
    """
    entries = []

    for class_name, paths in all_images.items():
        label        = 0 if class_name == "NORMAL" else 1
        manip        = "normal" if label == 0 else "abnormal"

        random.shuffle(paths)
        n         = len(paths)
        train_end = int(0.70 * n)
        val_end   = int(0.85 * n)

        split_map = {
            "train": paths[:train_end],
            "val":   paths[train_end:val_end],
            "test":  paths[val_end:],
        }

        for split_name, split_paths in split_map.items():
            dest_dir = os.path.join(processed_base, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            for src in tqdm(split_paths, desc=f"{split_name}/{class_name}"):
                fname = os.path.basename(src)
                dest  = os.path.join(dest_dir, fname)
                shutil.copy2(src, dest)

                entries.append({
                    "image_path":       dest,
                    "label":            label,
                    "dataset":          "APTOS",
                    "manipulation_type": manip,
                    "video_id":         os.path.splitext(fname)[0],
                })

        print(f"  [{class_name}] train={len(split_map['train'])}, "
              f"val={len(split_map['val'])}, test={len(split_map['test'])}")

    return entries

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n=== Collecting raw images ===")
    all_images = collect_images(RAW_BASE)
    print(f"\nTotal NORMAL  : {len(all_images['NORMAL'])}")
    print(f"Total ABNORMAL: {len(all_images['ABNORMAL'])}")

    print("\n=== Creating processed split ===")
    if os.path.exists(PROCESSED_BASE):
        print(f"  [WARN] {PROCESSED_BASE} already exists. Delete it first to re-run.")
    else:
        entries = create_split(all_images, PROCESSED_BASE)

        df = pd.DataFrame(entries)
        df.to_csv(CSV_SAVE_PATH, index=False)
        print(f"\n=== Metadata saved: {CSV_SAVE_PATH} ({len(df)} rows) ===")

        print("\nSplit summary:")
        for split in ["train", "val", "test"]:
            sub = df[df["image_path"].str.contains(f"/{split}/")]
            n0  = (sub["label"] == 0).sum()
            n1  = (sub["label"] == 1).sum()
            print(f"  {split:6s}: {n0} NORMAL, {n1} ABNORMAL")