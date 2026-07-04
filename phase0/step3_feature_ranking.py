import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    "raw_features.csv"
)

X = df.drop(
    columns=[
        "image",
        "emotion"
    ]
)

le = LabelEncoder()

y = le.fit_transform(
    df["emotion"]
)

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X,y)

importance = pd.DataFrame({

    "feature":
        X.columns,

    "importance":
        rf.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print(importance)

importance.to_csv(
    "feature_importance.csv",
    index=False
)