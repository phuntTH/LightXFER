import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

# Set up system path to find internal packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robust_data.dataset import create_dataloaders, FEATURE_COLUMNS
from multitask.multitask_model import MultiTaskFER

# Base directory navigation (Stepping out of 'training' folder to project root 'd:\LightXFER')
TRAINING_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)

# Correct paths pointing directly to project root
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "splits")
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")

BATCH_SIZE = 64
MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_NAMES = [f.replace("_", " ").title() for f in FEATURE_COLUMNS]

def run_threshold_optimization():
    print(f"🚀 Starting threshold optimization on device: {DEVICE.upper()}")
    print(f"📁 Target data directory: {SPLITS_DIR}")
    
    model = MultiTaskFER()
    model = model.to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Cannot find model weight file at {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    # Load dataloaders using the corrected root paths
    loaders = create_dataloaders(
        train_csv=TRAIN_CSV, val_csv=VAL_CSV, test_csv=TEST_CSV, batch_size=BATCH_SIZE
    )
    val_loader = loaders[1]  # Index 1 corresponds to validation loader

    all_raw_probs = []
    all_true_labels = []

    print("\n🔍 Running inference on validation set to collect raw probabilities...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Optimization Processing"):
            images = batch["image"].to(DEVICE, non_blocking=True)
            feature_labels = batch["features"].to(DEVICE, non_blocking=True)

            outputs = model(images)
            feature_out = outputs["feature_logits"]
            
            probs = torch.sigmoid(feature_out)

            all_raw_probs.append(probs.cpu().numpy())
            all_true_labels.append(feature_labels.cpu().numpy())

    all_raw_probs = np.vstack(all_raw_probs)    # Shape: (N, 8)
    all_true_labels = np.vstack(all_true_labels)  # Shape: (N, 8)

    adaptive_thresholds = {}

    print("\n📈 Optimizing threshold per feature column based on maximum F1-Score...")
    print("=" * 65)
    for i, name in enumerate(FEATURE_NAMES):
        y_true = all_true_labels[:, i]
        y_prob = all_raw_probs[:, i]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresholds):
            best_thresh = float(thresholds[best_idx])
        else:
            best_thresh = 0.50
            
        best_thresh = max(0.15, min(best_thresh, 0.65))
        adaptive_thresholds[name] = round(best_thresh, 4)
        print(f" 🔹 {name:20} -> Optimal Threshold: {best_thresh:.4f} (Max F1: {f1_scores[best_idx]:.4f})")
    print("=" * 65)

    output_json_path = os.path.join("checkpoints", "adaptive_thresholds.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(adaptive_thresholds, f, indent=4, ensure_ascii=False)
        
    print(f"\n💾 Successfully saved adaptive thresholds configuration to: {output_json_path}\n")

if __name__ == "__main__":
    run_threshold_optimization()