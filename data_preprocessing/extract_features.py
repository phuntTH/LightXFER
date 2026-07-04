import gc
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

METADATA_CSV = "dataset_final/metadata.csv"

OUTPUT_CSV = "data_preprocessing/all_features.csv"

mp_face_mesh = mp.solutions.face_mesh

# =====================================================
# FEATURE EXTRACTION
# =====================================================


def calculate_geometric_ratios(image_path, face_mesh):
    try:
        image = cv2.imread(image_path)

        if image is None:
            return None

        h, w = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False

        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        def get_dist(p1_idx, p2_idx):
            return np.linalg.norm(points[p1_idx] - points[p2_idx])

        # =====================================
        # Face normalization
        # =====================================

        face_width = get_dist(234, 454)

        face_height = get_dist(10, 152)

        if face_width < 1e-6:
            return None

        # =====================================
        # Mouth
        # =====================================

        mouth_width_ratio = get_dist(61, 291) / face_width

        mouth_open_ratio = get_dist(13, 14) / face_width

        mouth_aspect_ratio = mouth_open_ratio / (mouth_width_ratio + 1e-6)

        # =====================================
        # Eyes
        # =====================================

        left_eye_ratio = get_dist(159, 145) / face_width

        right_eye_ratio = get_dist(386, 374) / face_width

        eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # =====================================
        # Brows
        # =====================================

        brow_distance_ratio = get_dist(105, 334) / face_width

        left_brow_eye_ratio = get_dist(105, 159) / face_width

        right_brow_eye_ratio = get_dist(334, 386) / face_width

        # =====================================
        # Face shape
        # =====================================

        face_ratio = face_height / (face_width + 1e-6)

        return {
            "mouth_width_ratio": mouth_width_ratio,
            "mouth_open_ratio": mouth_open_ratio,
            "mouth_aspect_ratio": mouth_aspect_ratio,
            "left_eye_ratio": left_eye_ratio,
            "right_eye_ratio": right_eye_ratio,
            "eye_open_ratio": eye_open_ratio,
            "brow_distance_ratio": brow_distance_ratio,
            "left_brow_eye_ratio": left_brow_eye_ratio,
            "right_brow_eye_ratio": right_brow_eye_ratio,
            "face_ratio": face_ratio,
        }

    except Exception:
        return None


# =====================================================
# MAIN
# =====================================================


def extract_features():
    if not os.path.exists(METADATA_CSV):
        print(f"Metadata not found: {METADATA_CSV}")
        return

    metadata_df = pd.read_csv(METADATA_CSV)

    print(f"Loaded metadata: {len(metadata_df)} images")

    feature_records = []

    success_count = 0
    fail_count = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for _, row in tqdm(
            metadata_df.iterrows(),
            total=len(metadata_df),
            desc="Extracting Features",
        ):
            image_path = row["image_path"]

            ratios = calculate_geometric_ratios(
                image_path=image_path, face_mesh=face_mesh
            )

            if ratios is None:
                fail_count += 1
                continue

            feature_row = {
                "image_path": image_path,
                "emotion": row["emotion"],
                "source": row["source"],
                "original_file": row["original_file"],
            }

            feature_row.update(ratios)

            feature_records.append(feature_row)

            success_count += 1

        gc.collect()

    feature_df = pd.DataFrame(feature_records)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    feature_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)

    print(f"Total Images : {len(metadata_df)}")

    print(f"Success      : {success_count}")

    print(f"Failed       : {fail_count}")

    print(f"Saved CSV    : {OUTPUT_CSV}")

    print(f"Rows         : {len(feature_df)}")

    print("=" * 60)


if __name__ == "__main__":
    extract_features()