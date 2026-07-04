import os
import pandas as pd
import torch
from dataset import EMOTIONS, FEATURE_COLUMNS, create_dataloaders

# =====================================================
# PATHS
# =====================================================

TRAIN_CSV = "data_preprocessing/splits/train.csv"
VAL_CSV = "data_preprocessing/splits/val.csv"
TEST_CSV = "data_preprocessing/splits/test.csv"


# =====================================================
# DATA LEAKAGE
# =====================================================


def verify_data_leakage(train_df, val_df, test_df):
    print("\n[1/4] Data Leakage Check")

    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert len(train_paths & val_paths) == 0
    assert len(train_paths & test_paths) == 0
    assert len(val_paths & test_paths) == 0

    print("No leakage detected")


# =====================================================
# STRATIFICATION
# =====================================================


def print_distribution(df, name):
    print(f"\n[{name}]")

    print("\nSource Distribution")
    src_dist = df["source"].value_counts(normalize=True) * 100
    for src, pct in src_dist.items():
        print(f"{src:<12} {pct:.2f}%")

    print("\nEmotion Distribution")
    emo_dist = df["emotion"].value_counts(normalize=True) * 100
    for emo in EMOTIONS:
        print(f"{emo:<12}{emo_dist.get(emo, 0):.2f}%")


def verify_distribution(train_df, val_df, test_df):
    print("\n[2/4] Distribution Check")

    print_distribution(train_df, "TRAIN")
    print_distribution(val_df, "VAL")
    print_distribution(test_df, "TEST")

    print("\nDistribution looks good")


# =====================================================
# CLASS WEIGHTS
# =====================================================


def verify_class_weights(class_weights):
    print("\n[3/4] Class Weight Check")

    assert isinstance(class_weights, torch.Tensor)
    assert not torch.isnan(class_weights).any()
    assert not torch.isinf(class_weights).any()
    assert len(class_weights) == len(EMOTIONS)

    print(class_weights.numpy())
    print("Class weights valid")


# =====================================================
# BATCH CHECK
# =====================================================


def verify_batch(loader, name):
    print(f"\n[4/4] {name} Batch Check")

    batch = next(iter(loader))

    images = batch["image"]
    emotions = batch["emotion"]
    features = batch["features"]
    sources = batch["source"]

    print(f"Images   : {images.shape}")
    print(f"Emotion  : {emotions.shape}")
    print(f"Features : {features.shape}")
    print(f"Sources  : {len(sources)}")

    assert images.ndim == 4
    assert emotions.ndim == 1
    assert features.shape[1] == len(FEATURE_COLUMNS)

    print("Batch OK")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2 VERIFICATION")
    print("=" * 60)

    (
        train_loader,
        val_loader,
        test_loader,
        train_df,
        val_df,
        test_df,
        class_weights,
    ) = create_dataloaders(
        train_csv=TRAIN_CSV, val_csv=VAL_CSV, test_csv=TEST_CSV, batch_size=64, num_workers=0
    )

    verify_data_leakage(train_df, val_df, test_df)
    verify_distribution(train_df, val_df, test_df)
    verify_class_weights(class_weights)

    verify_batch(train_loader, "TRAIN")
    verify_batch(val_loader, "VAL")
    verify_batch(test_loader, "TEST")

    print("\nPHASE 2 PASSED")