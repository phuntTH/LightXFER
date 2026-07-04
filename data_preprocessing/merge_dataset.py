import os
import shutil
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

SOURCES = ["data/fer2013", "data/affectnet", "data/raf_db"]

TARGET_DIR = "dataset_final"

METADATA_CSV = os.path.join(TARGET_DIR, "metadata.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# =====================================================
# LABEL MAP
# =====================================================

LABEL_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise",
    # AffectNet
    "contempt": None,
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# =====================================================
# COPY
# =====================================================


def copy_images_and_build_metadata():
    metadata_rows = []
    stats = defaultdict(int)
    global_counter = 0

    for dataset_path in SOURCES:
        dataset_name = os.path.basename(dataset_path)

        print("\n" + "=" * 60)
        print(f"Processing: {dataset_name}")
        print("=" * 60)

        for split in ["train", "test"]:
            split_dir = os.path.join(dataset_path, split)

            if not os.path.exists(split_dir):
                continue

            for emotion_folder in os.listdir(split_dir):
                mapped_emotion = LABEL_MAP.get(emotion_folder.lower())

                if mapped_emotion is None:
                    continue

                src_emotion_dir = os.path.join(split_dir, emotion_folder)

                if not os.path.isdir(src_emotion_dir):
                    continue

                dst_emotion_dir = os.path.join(
                    TARGET_DIR, split, mapped_emotion
                )

                os.makedirs(dst_emotion_dir, exist_ok=True)

                image_files = [
                    f
                    for f in os.listdir(src_emotion_dir)
                    if f.lower().endswith(IMG_EXTS)
                ]

                for image_file in tqdm(
                    image_files,
                    desc=f"{dataset_name} | {split} | {mapped_emotion}",
                    leave=False,
                ):
                    src_path = os.path.join(src_emotion_dir, image_file)
                    ext = os.path.splitext(image_file)[1]

                    global_counter += 1

                    new_name = (
                        f"{dataset_name}_" f"{global_counter:07d}" f"{ext}"
                    )

                    dst_path = os.path.join(dst_emotion_dir, new_name)

                    shutil.copy2(src_path, dst_path)

                    relative_path = os.path.join(
                        TARGET_DIR, split, mapped_emotion, new_name
                    ).replace("\\", "/")

                    metadata_rows.append(
                        {
                            "image_path": relative_path,
                            "emotion": mapped_emotion,
                            "source": dataset_name,
                            "original_file": image_file,
                        }
                    )

                    stats[(split, mapped_emotion)] += 1

    return metadata_rows, stats


# =====================================================
# SAVE CSV
# =====================================================


def save_metadata(metadata_rows):
    df = pd.DataFrame(metadata_rows)

    df.to_csv(METADATA_CSV, index=False)

    print("\n" + "=" * 60)
    print("METADATA SAVED")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"File: {METADATA_CSV}")

    print("\nSource Distribution")
    print(df["source"].value_counts())

    print("\nEmotion Distribution")
    print(df["emotion"].value_counts())


# =====================================================
# STATS
# =====================================================


def print_stats(stats):
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)

    for split in ["train", "test"]:
        print(f"\n[{split.upper()}]")

        total = 0

        for emotion in EMOTIONS:
            count = stats[(split, emotion)]
            total += count
            print(f"{emotion:<12}: {count}")

        print(f"Total       : {total}")


# =====================================================
# MAIN
# =====================================================


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    metadata_rows, stats = copy_images_and_build_metadata()

    save_metadata(metadata_rows)
    print_stats(stats)

    print("\nMerge Complete!")


if __name__ == "__main__":
    main()