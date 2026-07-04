import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import f_oneway

df = pd.read_csv("raw_features.csv")

FEATURES = [
    col
    for col in df.columns
    if col not in ["image", "emotion"]
]

print("=" * 80)
print("ANOVA TEST")
print("=" * 80)

for feature in FEATURES:

    groups = []

    for emotion in sorted(
        df["emotion"].unique()
    ):

        values = df[
            df["emotion"] == emotion
        ][feature]

        groups.append(values)

    f_stat, p_value = f_oneway(
        *groups
    )

    print(
        f"{feature:<25}"
        f" p-value = {p_value:.10f}"
    )

print("\nDone.")

# --------------------------------------------------
# Boxplot
# --------------------------------------------------

os.makedirs(
    "plots",
    exist_ok=True
)

for feature in FEATURES:

    plt.figure(figsize=(10,5))

    sns.boxplot(
        data=df,
        x="emotion",
        y=feature
    )

    plt.title(feature)

    plt.tight_layout()

    plt.savefig(
        f"plots/{feature}.png"
    )

    plt.close()

print("Plots saved.")