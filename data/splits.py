import os
import random

random.seed(42)


def get_splits(dataset_name, raw_path):

    if dataset_name == "FFPP":
        return get_ffpp_splits(raw_path)

    elif dataset_name == "CelebDF":
        return get_celebdf_splits(raw_path)

    elif dataset_name == "DFD":
        return get_dfd_splits(raw_path)
    
    elif dataset_name == "APTOS":
        return get_aptos_splits(raw_path)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_ffpp_splits(raw_path):
    """
    Split by real video ID to prevent data leakage.
    FFPP fake video IDs are like "001_003" where "001" is the source real video.
    We assign each fake to the same split as its source real video.
    """
    splits = {"train": [], "val": [], "test": []}

    original_path = os.path.join(
        raw_path,
        "original_sequences/youtube/c23/videos"
    )

    manip_types = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter"
    ]

    # --- Load real videos and split by ID ---
    real_video_paths = []
    for vid in os.listdir(original_path):
        if vid.endswith(".mp4"):
            real_video_paths.append({
                "path": os.path.join(original_path, vid),
                "label": 0,
                "manipulation": "real",
                "video_id": vid.replace(".mp4", "")
            })

    # Shuffle real video IDs to assign splits
    random.shuffle(real_video_paths)
    n = len(real_video_paths)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    train_ids = set(v["video_id"] for v in real_video_paths[:train_end])
    val_ids   = set(v["video_id"] for v in real_video_paths[train_end:val_end])
    test_ids  = set(v["video_id"] for v in real_video_paths[val_end:])

    # Assign real videos to splits
    for v in real_video_paths[:train_end]:
        splits["train"].append(v)
    for v in real_video_paths[train_end:val_end]:
        splits["val"].append(v)
    for v in real_video_paths[val_end:]:
        splits["test"].append(v)

    # --- Load fake videos and assign to the same split as their source ---
    for manip in manip_types:
        manip_path = os.path.join(
            raw_path,
            f"manipulated_sequences/{manip}/c23/videos"
        )

        for vid in os.listdir(manip_path):
            if not vid.endswith(".mp4"):
                continue

            video_id = vid.replace(".mp4", "")
            # FFPP fake IDs look like "001_003" — source is the first part
            source_id = video_id.split("_")[0]

            video_info = {
                "path": os.path.join(manip_path, vid),
                "label": 1,
                "manipulation": manip,
                "video_id": video_id
            }

            if source_id in train_ids:
                splits["train"].append(video_info)
            elif source_id in val_ids:
                splits["val"].append(video_info)
            elif source_id in test_ids:
                splits["test"].append(video_info)
            else:
                # Fallback: source ID not found (e.g. FaceShifter uses different naming),
                # assign proportionally to avoid losing data
                splits["train"].append(video_info)

    print(f"[FFPP] Train: {len(splits['train'])}, "
          f"Val: {len(splits['val'])}, "
          f"Test: {len(splits['test'])}")

    return splits


def get_celebdf_splits(raw_path):
    """
    Use the official List_of_testing_videos.txt for the test set.
    The file format is: "<label> <folder>/<filename>.mp4"
    e.g. "1 Celeb-synthesis/id45_id52_0001.mp4"
    """
    splits = {"train": [], "val": [], "test": []}

    test_list_path = os.path.join(raw_path, "List_of_testing_videos.txt")

    # Parse the test list — extract just the filename (not folder prefix)
    test_video_filenames = set()
    with open(test_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                # parts[1] is like "Celeb-synthesis/id45_id52_0001.mp4"
                filename = parts[1].split("/")[-1]
                test_video_filenames.add(filename)

    categories = {
        "Celeb-real": 0,
        "YouTube-real": 0,
        "Celeb-synthesis": 1
    }

    all_train_videos = []

    for folder, label in categories.items():
        folder_path = os.path.join(raw_path, folder)

        if not os.path.exists(folder_path):
            print(f"[CelebDF] Warning: folder not found: {folder_path}")
            continue

        for vid in os.listdir(folder_path):
            if not vid.endswith(".mp4"):
                continue

            video_info = {
                "path": os.path.join(folder_path, vid),
                "label": label,
                "manipulation": "fake" if label == 1 else "real",
                "video_id": vid.replace(".mp4", "")
            }

            if vid in test_video_filenames:
                splits["test"].append(video_info)
            else:
                all_train_videos.append(video_info)

    random.shuffle(all_train_videos)
    n = len(all_train_videos)
    train_end = int(0.85 * n)

    splits["train"] = all_train_videos[:train_end]
    splits["val"]   = all_train_videos[train_end:]

    print(f"[CelebDF] Train: {len(splits['train'])}, "
          f"Val: {len(splits['val'])}, "
          f"Test: {len(splits['test'])}")

    if len(splits["test"]) == 0:
        print("[CelebDF] WARNING: Test split is empty! "
              "Check that List_of_testing_videos.txt filenames match your video files.")

    return splits


def get_dfd_splits(raw_path):

    splits = {"train": [], "val": [], "test": []}

    fake_root = os.path.join(
        raw_path,
        "DFD_manipulated_sequences",
        "DFD_manipulated_sequences"
    )

    real_root = os.path.join(
        raw_path,
        "DFD_original_sequences"
    )

    video_entries = []

    if os.path.exists(fake_root):
        for vid in os.listdir(fake_root):
            if vid.endswith(".mp4"):
                video_entries.append({
                    "path": os.path.join(fake_root, vid),
                    "label": 1,
                    "manipulation": "fake",
                    "video_id": vid.replace(".mp4", "")
                })

    if os.path.exists(real_root):
        for vid in os.listdir(real_root):
            if vid.endswith(".mp4"):
                video_entries.append({
                    "path": os.path.join(real_root, vid),
                    "label": 0,
                    "manipulation": "real",
                    "video_id": vid.replace(".mp4", "")
                })

    random.shuffle(video_entries)

    n = len(video_entries)
    train_end = int(0.80 * n)
    val_end   = int(0.90 * n)

    splits["train"] = video_entries[:train_end]
    splits["val"]   = video_entries[train_end:val_end]
    splits["test"]  = video_entries[val_end:]

    print(f"[DFD] Loaded {n} videos")
    print(f"[DFD] Train: {len(splits['train'])}, "
          f"Val: {len(splits['val'])}, "
          f"Test: {len(splits['test'])}")

    return splits

def get_aptos_splits(raw_path):
    """
    For APTOS, the split is already done by prepare_dataset.py.
    This function reads from the processed folder structure.
    Used by preprocessing.py to iterate over already-processed images.
    """
    splits = {"train": [], "val": [], "test": []}
    class_map = {"NORMAL": 0, "ABNORMAL": 1}

    for split_name in ["train", "val", "test"]:
        for class_name, label in class_map.items():
            folder = os.path.join(raw_path, split_name, class_name)
            if not os.path.exists(folder):
                continue
            for img in sorted(os.listdir(folder)):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    splits[split_name].append({
                        "path":         os.path.join(folder, img),
                        "label":        label,
                        "manipulation": "normal" if label == 0 else "abnormal",
                        "video_id":     os.path.splitext(img)[0],
                    })

    for s in ["train", "val", "test"]:
        n0 = sum(1 for v in splits[s] if v["label"] == 0)
        n1 = sum(1 for v in splits[s] if v["label"] == 1)
        print(f"[APTOS] {s:6s}: {n0} NORMAL, {n1} ABNORMAL")

    return splits