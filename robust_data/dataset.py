import numpy as np
import pandas as pd
import cv2

cv2.setNumThreads(0)

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =====================================================
# CONSTANTS
# =====================================================

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

FEATURE_COLUMNS = [
    "smile",
    "mouth_open",
    "eyes_wide_open",
    "eyes_narrowed",
    "brows_together",
    "lowered_eyebrows",
    "raised_eyebrows",
    "compressed_lips",
]


# =====================================================
# DATASET
# =====================================================


class FERDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        emotion = EMOTION_TO_IDX[row["emotion"]]

        features = torch.tensor(
            row[FEATURE_COLUMNS].values.astype(np.float32), dtype=torch.float32
        )

        source = row["source"]

        return {
            "image": image,
            "emotion": emotion,
            "features": features,
            "source": source,
        }


# =====================================================
# ALBUMENTATIONS
# =====================================================


def get_train_transform():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    )


def get_eval_transform():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    )


# =====================================================
# CLASS WEIGHTS
# =====================================================


def compute_class_weights(train_df):
    class_counts = train_df["emotion"].value_counts()

    counts = np.array([class_counts.get(emotion, 1) for emotion in EMOTIONS])

    total_samples = len(train_df)

    class_weights = total_samples / (len(EMOTIONS) * counts)

    return torch.tensor(class_weights, dtype=torch.float32)


# =====================================================
# DATALOADER FACTORY
# =====================================================


def create_dataloaders(
    train_csv, val_csv, test_csv, batch_size=64, num_workers=2
):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_dataset = FERDataset(train_df, transform=get_train_transform())

    val_dataset = FERDataset(val_df, transform=get_eval_transform())

    test_dataset = FERDataset(test_df, transform=get_eval_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    class_weights = compute_class_weights(train_df)

    total_samples = len(train_df) + len(val_df) + len(test_df)

    print("\n" + "=" * 60)
    print("FIXED-SPLIT DATA PIPELINE INITIALIZED")
    print("=" * 60)

    print(
        f"Train Samples : {len(train_df):,}"
        f" ({len(train_df)/total_samples*100:.1f}%)"
    )

    print(
        f"Val Samples   : {len(val_df):,}"
        f" ({len(val_df)/total_samples*100:.1f}%)"
    )

    print(
        f"Test Samples  : {len(test_df):,}"
        f" ({len(test_df)/total_samples*100:.1f}%)"
    )

    print("\nClass Weights:")
    print(class_weights.numpy())
    print("=" * 60)

    return (
        train_loader,
        val_loader,
        test_loader,
        train_df,
        val_df,
        test_df,
        class_weights,
    )
