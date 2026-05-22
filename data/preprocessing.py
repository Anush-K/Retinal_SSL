import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
from data.face_extract import FaceExtractor
from data.splits import get_splits
from data.metadata import MetadataWriter


def sample_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def process_dataset(dataset_name, raw_path, processed_path, debug=False):
    face_extractor = None
    if dataset_name != "APTOS":
        from data.face_extract import FaceExtractor
        face_extractor = FaceExtractor(target_size=224)
    
    splits = get_splits(dataset_name, raw_path)

    if dataset_name == "APTOS":
        csv_save_path = "/content/drive/MyDrive/Retinal_SSL/APTOS_metadata.csv"
    else:
        csv_save_path = os.path.join(processed_path, f"{dataset_name}_metadata.csv")

    metadata_writer = MetadataWriter(save_path=csv_save_path)

    total_saved = 0
    total_skipped = 0

    for split_name, video_list in splits.items():

        if debug:
            video_list = video_list[:3]

        for video_info in tqdm(video_list, desc=f"{dataset_name}/{split_name}"):
            video_path   = video_info["path"]
            label        = video_info["label"]
            manipulation = video_info.get("manipulation", "NA")
            video_id     = video_info["video_id"]

            # ── APTOS: images already processed, just register metadata ──
            if dataset_name == "APTOS":
                if os.path.exists(video_path):
                    metadata_writer.add_entry(
                        image_path   = video_path,
                        label        = label,
                        dataset      = dataset_name,
                        manipulation = manipulation,
                        video_id     = video_id,
                    )
                    total_saved += 1
                else:
                    print(f"  [WARN] Not found: {video_path}")
                continue

            # ── Video datasets: sample frames and extract faces ──
            frames = sample_frames(video_path, num_frames=8)

            for i, frame in enumerate(frames):
                face = face_extractor.extract_face(frame)

                if face is None:
                    total_skipped += 1
                    continue

                save_dir = os.path.join(
                    processed_path, dataset_name, split_name,
                    "real" if label == 0 else "fake"
                )
                os.makedirs(save_dir, exist_ok=True)

                # filename  = f"{video_id}_{i}.jpg"
                filename = f"{manipulation}_{video_id}_{i}.jpg"
                save_path = os.path.join(save_dir, filename)

                cv2.imwrite(save_path, face, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                metadata_writer.add_entry(
                    image_path=save_path,
                    label=label,
                    dataset=dataset_name,
                    manipulation=manipulation,
                    video_id=video_id
                )
                total_saved += 1

    metadata_writer.save()
    print(f"\n[{dataset_name}] Done. Saved: {total_saved} faces, "
          f"Skipped (no face detected): {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["FFPP", "CelebDF", "DFD", "APTOS"],
                        help="Which dataset to process")
    parser.add_argument("--debug", action="store_true",
                        help="Process only 3 items per split for quick testing")
    args = parser.parse_args()

    # ── Path configuration ────────────────────────────────────
    if args.dataset == "APTOS":
        RAW_BASE       = "/content/drive/MyDrive"
        PROCESSED_BASE = "/content/drive/MyDrive"
        folder_map     = {"APTOS": "APTOS_processed"}
    else:
        RAW_BASE       = "/content/drive/MyDrive/DF_Datasets"
        PROCESSED_BASE = "/content/drive/MyDrive/DF_Datasets/processed"
        folder_map     = {
            "FFPP":    "FFPP_raw",
            "CelebDF": "CelebDF_raw",
            "DFD":     "DFD_raw",
        }

    raw_path = os.path.join(RAW_BASE, folder_map[args.dataset])
    os.makedirs(PROCESSED_BASE, exist_ok=True)

    print(f"Processing dataset : {args.dataset}")
    print(f"Raw path           : {raw_path}")
    print(f"Output path        : {PROCESSED_BASE}")
    print(f"Debug mode         : {args.debug}\n")

    process_dataset(args.dataset, raw_path, PROCESSED_BASE, debug=args.debug)