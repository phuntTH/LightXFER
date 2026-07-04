import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = r"D:\FER\data\fer2013\train"

OUTPUT_CSV = "raw_features.csv"

# =====================================================
# MEDIAPIPE
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# =====================================================
# LANDMARK UTILS
# =====================================================

def euclidean_distance(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def extract_landmarks(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = []

    for lm in results.multi_face_landmarks[0].landmark:

        landmarks.append(
            [lm.x, lm.y]
        )

    return landmarks

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(landmarks):

    # --------------------------
    # Face Size
    # --------------------------

    face_width = euclidean_distance(
        landmarks[234],
        landmarks[454]
    )

    face_height = euclidean_distance(
        landmarks[10],
        landmarks[152]
    )

    if face_width < 1e-6:
        return None

    if face_height < 1e-6:
        return None

    # --------------------------
    # Mouth
    # --------------------------

    mouth_width = euclidean_distance(
        landmarks[61],
        landmarks[291]
    )

    mouth_height = euclidean_distance(
        landmarks[13],
        landmarks[14]
    )

    mouth_width_ratio = (
        mouth_width / face_width
    )

    mouth_height_ratio = (
        mouth_height / face_height
    )

    mouth_aspect_ratio = (
        mouth_height /
        (mouth_width + 1e-6)
    )

    # --------------------------
    # Eyes
    # --------------------------

    left_eye_height = euclidean_distance(
        landmarks[159],
        landmarks[145]
    )

    right_eye_height = euclidean_distance(
        landmarks[386],
        landmarks[374]
    )

    left_eye_ratio = (
        left_eye_height /
        face_height
    )

    right_eye_ratio = (
        right_eye_height /
        face_height
    )

    avg_eye_ratio = (
        left_eye_ratio +
        right_eye_ratio
    ) / 2

    # --------------------------
    # Eyebrows
    # --------------------------

    brow_distance = euclidean_distance(
        landmarks[105],
        landmarks[334]
    )

    brow_distance_ratio = (
        brow_distance /
        face_width
    )

    left_brow_eye = euclidean_distance(
        landmarks[105],
        landmarks[159]
    )

    right_brow_eye = euclidean_distance(
        landmarks[334],
        landmarks[386]
    )

    left_brow_eye_ratio = (
        left_brow_eye /
        face_height
    )

    right_brow_eye_ratio = (
        right_brow_eye /
        face_height
    )

    # --------------------------
    # Face Shape
    # --------------------------

    face_ratio = (
        face_height /
        face_width
    )

    return {

        "mouth_width_ratio":
            mouth_width_ratio,

        "mouth_height_ratio":
            mouth_height_ratio,

        "mouth_aspect_ratio":
            mouth_aspect_ratio,

        "left_eye_ratio":
            left_eye_ratio,

        "right_eye_ratio":
            right_eye_ratio,

        "avg_eye_ratio":
            avg_eye_ratio,

        "brow_distance_ratio":
            brow_distance_ratio,

        "left_brow_eye_ratio":
            left_brow_eye_ratio,

        "right_brow_eye_ratio":
            right_brow_eye_ratio,

        "face_ratio":
            face_ratio
    }

# =====================================================
# DATASET SCAN
# =====================================================

rows = []

emotion_folders = sorted(
    os.listdir(DATA_DIR)
)

for emotion in emotion_folders:

    emotion_dir = os.path.join(
        DATA_DIR,
        emotion
    )

    if not os.path.isdir(
        emotion_dir
    ):
        continue

    print(
        f"\nProcessing {emotion}"
    )

    image_files = os.listdir(
        emotion_dir
    )

    for filename in tqdm(image_files):

        image_path = os.path.join(
            emotion_dir,
            filename
        )

        landmarks = extract_landmarks(
            image_path
        )

        if landmarks is None:
            continue

        features = extract_features(
            landmarks
        )

        if features is None:
            continue

        row = {

            "image":
                image_path,

            "emotion":
                emotion
        }

        row.update(features)

        rows.append(row)

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nDone!")

print(
    f"Saved: {OUTPUT_CSV}"
)

print(
    f"Samples: {len(df)}"
)

print("\nColumns:")

print(df.columns.tolist())

print("\nFirst rows:")

print(df.head())