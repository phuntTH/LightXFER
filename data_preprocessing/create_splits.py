import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "data_preprocessing/pseudo_labels.csv"
OUTPUT_DIR = "data_preprocessing/splits"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_STATE = 42


def create_splits():
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} samples")

    # =====================================
    # Dual Stratification
    # =====================================
    df["stratify_key"] = df["emotion"].astype(str) + "_" + df["source"].astype(str)

    # =====================================
    # Train (80%)
    # Temp  (20%)
    # =====================================
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=df["stratify_key"],
        random_state=RANDOM_STATE,
    )

    # =====================================
    # Val (10%)
    # Test (10%)
    # =====================================
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["stratify_key"],
        random_state=RANDOM_STATE,
    )

    # =====================================
    # Cleanup
    # =====================================
    train_df = train_df.drop(columns=["stratify_key"])
    val_df = val_df.drop(columns=["stratify_key"])
    test_df = test_df.drop(columns=["stratify_key"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("SPLIT COMPLETED")
    print("=" * 60)
    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")
    print("\nSaved:")
    print(train_path)
    print(val_path)
    print(test_path)
    print("=" * 60)


if __name__ == "__main__":
    create_splits()