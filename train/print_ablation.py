"""
print_ablation.py
Loads all saved result CSVs and prints the LSFC ablation table.
"""

import pandas as pd
import os

RESULTS_BASE = "/content/drive/MyDrive/Retinal_SSL"

runs = [
    ("No SSL (ImageNet baseline)",                  "results_nossl"),
    ("SSL Spatial-Only (plain SimCLR)",              "results_spatial_only"),
    ("SSL Single-Band (DBFC replica)",               "results_single_band"),
    ("SSL Multi-Band, naive (shared global pool)",   "results_multi_band"),
    ("SSL Multi-Band, scale-preserving (LSFC — Ours)", "results_multi_band_sp"),
]

rows = []
for label, folder in runs:
    csv_path = os.path.join(RESULTS_BASE, folder, "evaluation_summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df.iloc[0].to_dict()
        row["Run"] = label
        rows.append(row)
    else:
        print(f"  [MISSING] {csv_path}")

if rows:
    ablation_df = pd.DataFrame(rows)
    cols = ["Run", "AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1"]
    ablation_df = ablation_df[cols]

    print("\n" + "="*100)
    print("ABLATION TABLE — APTOS Retinal Image Classification (LSFC)")
    print("="*100)
    print(ablation_df.to_string(index=False))
    print("="*100)

    ablation_df.to_csv(os.path.join(RESULTS_BASE, "ablation_table.csv"), index=False)
    print("\nSaved: ablation_table.csv")