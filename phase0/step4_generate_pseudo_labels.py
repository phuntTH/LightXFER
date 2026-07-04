import pandas as pd

df = pd.read_csv(
    "raw_features.csv"
)

# --------------------------------------------------
# Smile
# --------------------------------------------------

smile_threshold = df[
    "mouth_width_ratio"
].quantile(0.75)

df["smile"] = (
    df["mouth_width_ratio"]
    >
    smile_threshold
).astype(int)

# --------------------------------------------------
# Mouth Open
# --------------------------------------------------

mouth_open_threshold = df[
    "mouth_height_ratio"
].quantile(0.75)

df["mouth_open"] = (
    df["mouth_height_ratio"]
    >
    mouth_open_threshold
).astype(int)

# --------------------------------------------------
# Eyes Wide
# --------------------------------------------------

eye_threshold = df[
    "avg_eye_ratio"
].quantile(0.75)

df["eyes_wide"] = (
    df["avg_eye_ratio"]
    >
    eye_threshold
).astype(int)

# --------------------------------------------------
# Brows Together
# --------------------------------------------------

brow_threshold = df[
    "brow_distance_ratio"
].quantile(0.25)

df["brows_together"] = (
    df["brow_distance_ratio"]
    <
    brow_threshold
).astype(int)

df.to_csv(
    "pseudo_labels_v2.csv",
    index=False
)

print(
    df[
        [
            "emotion",
            "smile",
            "mouth_open",
            "eyes_wide",
            "brows_together"
        ]
    ].head()
)

print(
    "\nSmile Correlation:\n"
)

print(
    pd.crosstab(
        df["emotion"],
        df["smile"],
        normalize="index"
    )
)