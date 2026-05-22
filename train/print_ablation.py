"""
print_ablation.py
Loads all saved result CSVs and prints a clean ablation table.
Run after all training conditions are complete.
"""

import pandas as pd
import os

RESULTS_BASE = "/content/drive/MyDrive/Retinal_SSL"

runs = [
    ("No SSL (ImageNet baseline)",          "results_nossl"),
    ("SSL Spatial Symmetric (SimCLR)",      "results_ssl_symmetric"),
    ("SSL Spatial Asymmetric",              "results_ssl_asymmetric"),
    ("DBFC SSL (Full — Ours)",              "results_ssl_dbfc"),
    ("DBFC SSL + Full Unfreeze",            "results_ssl_fullunfreeze"),
]

rows = []
for label, folder in runs:
    csv_path = os.path.join(RESULTS_BASE, folder, "evaluation_summary.csv")
    if os.path.exists(csv_path):
        df  = pd.read_csv(csv_path)
        row = df.iloc[0].to_dict()
        row["Run"] = label
        rows.append(row)
    else:
        print(f"  [MISSING] {csv_path}")

if rows:
    ablation_df = pd.DataFrame(rows)
    cols = ["Run", "AUC", "Accuracy", "Precision",
            "Recall", "Specificity", "F1"]
    ablation_df = ablation_df[cols]

    print("\n" + "="*90)
    print("ABLATION TABLE — APTOS Retinal Image Classification")
    print("="*90)
    print(ablation_df.to_string(index=False))
    print("="*90)

    ablation_df.to_csv(
        os.path.join(RESULTS_BASE, "ablation_table.csv"), index=False
    )
    print("\nSaved: ablation_table.csv")