import os
import pandas as pd

INPUT_CSV = "data_preprocessing/all_features.csv"
OUTPUT_CSV = "data_preprocessing/pseudo_labels.csv"


def generate_pseudo_labels():

    df = pd.read_csv(INPUT_CSV)

    print(f"Loaded {len(df)} samples")

    # =====================================
    # Neutral Baseline
    # =====================================

    neutral_df = df[
        df["emotion"] == "neutral"
    ]

    print(
        f"Neutral Samples: {len(neutral_df)}"
    )

    # =====================================
    # Thresholds
    # =====================================

    smile_thr = neutral_df[
        "mouth_width_ratio"
    ].quantile(0.95)

    mouth_open_thr = neutral_df[
        "mouth_open_ratio"
    ].quantile(0.97)

    eye_wide_thr = neutral_df[
        "eye_open_ratio"
    ].quantile(0.90)

    eye_narrow_thr = neutral_df[
        "eye_open_ratio"
    ].quantile(0.10)

    brow_together_thr = neutral_df[
        "brow_distance_ratio"
    ].quantile(0.15)

    avg_brow_eye_neutral = (
        neutral_df["left_brow_eye_ratio"]
        +
        neutral_df["right_brow_eye_ratio"]
    ) / 2

    lowered_brow_thr = (
        avg_brow_eye_neutral
        .quantile(0.15)
    )

    raised_brow_thr = (
        avg_brow_eye_neutral
        .quantile(0.85)
    )

    compressed_thr = neutral_df[
        "mouth_aspect_ratio"
    ].quantile(0.15)

    # =====================================
    # Average Brow Eye
    # =====================================

    avg_brow_eye = (
        df["left_brow_eye_ratio"]
        +
        df["right_brow_eye_ratio"]
    ) / 2

    # =====================================
    # Labels
    # =====================================

    df["smile"] = (
        df["mouth_width_ratio"]
        > smile_thr
    ).astype(int)

    df["mouth_open"] = (
        df["mouth_open_ratio"]
        > mouth_open_thr
    ).astype(int)

    df["eyes_wide_open"] = (
        df["eye_open_ratio"]
        > eye_wide_thr
    ).astype(int)

    df["eyes_narrowed"] = (
        df["eye_open_ratio"]
        < eye_narrow_thr
    ).astype(int)

    df["brows_together"] = (
        df["brow_distance_ratio"]
        < brow_together_thr
    ).astype(int)

    df["lowered_eyebrows"] = (
        avg_brow_eye
        < lowered_brow_thr
    ).astype(int)

    df["raised_eyebrows"] = (
        avg_brow_eye
        > raised_brow_thr
    ).astype(int)

    df["compressed_lips"] = (
        (df["mouth_aspect_ratio"] < compressed_thr)
        &
        (df["mouth_open"] == 0)
    ).astype(int)

    # =====================================
    # Save
    # =====================================

    output_df = df[
        [
            "image_path",
            "emotion",
            "source",

            "smile",
            "mouth_open",

            "eyes_wide_open",
            "eyes_narrowed",

            "brows_together",
            "lowered_eyebrows",
            "raised_eyebrows",

            "compressed_lips"
        ]
    ]

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nSaved:", OUTPUT_CSV)

    print("\nDistribution")

    for col in output_df.columns[3:]:

        pct = (
            output_df[col]
            .mean()
            * 100
        )

        print(
            f"{col:<20}"
            f"{pct:.2f}%"
        )


if __name__ == "__main__":
    generate_pseudo_labels()