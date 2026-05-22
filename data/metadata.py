import pandas as pd


class MetadataWriter:
    def __init__(self, save_path):
        self.save_path = save_path
        self.entries = []

    def add_entry(self, image_path, label, dataset, manipulation, video_id):
        self.entries.append({
            "image_path": image_path,
            "label": label,
            "dataset": dataset,
            "manipulation_type": manipulation,
            "video_id": video_id
        })

    def save(self):
        df = pd.DataFrame(self.entries)
        df.to_csv(self.save_path, index=False)
        print(f"Metadata saved to {self.save_path} ({len(df)} rows)")