# XAI-FER: Explainable Multi-Task Facial Emotion Recognition

XAI-FER is a multi-task facial emotion recognition system that combines emotion classification, facial attribute prediction, occlusion-aware training, and explainable AI techniques.

The project uses a shared `MobileNetV3-Large` backbone to predict facial emotions and landmark-derived facial attributes simultaneously. It is designed to improve robustness in real-world conditions, particularly when parts of the face such as the mouth are occluded by masks.

The system also integrates Grad-CAM visualizations to help analyze which facial regions contribute to each emotion prediction.

## Features

* Multi-task learning for facial emotion recognition and facial attribute prediction
* Seven-class emotion classification
* Eight facial attribute predictions generated from facial landmark geometry
* MobileNetV3-Large shared feature extractor
* Focal Loss for handling emotion class imbalance
* Binary Cross-Entropy loss for multi-label facial attributes
* Occlusion-aware loss masking for mouth-related facial features
* Weighted random sampling for imbalanced datasets
* Per-attribute threshold optimization based on validation F1-score
* Grad-CAM visualization for explainability
* Real-time webcam inference
* Evaluation reports, confusion matrices, and misclassified image analysis

## Supported Emotion Classes

The model predicts the following seven facial emotion classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## Facial Attribute Labels

In addition to emotion classification, the model predicts eight facial attributes:

* Raised Eyebrows
* Lowered Eyebrows
* Eyebrows Together
* Eyes Wide Open
* Eyes Narrowed
* Mouth Open
* Smile
* Compressed Lips

These attributes are generated as pseudo-labels from facial landmark geometry extracted with MediaPipe FaceMesh. They provide auxiliary supervision during multi-task training.

## System Architecture

```text
Input Face Image
        |
        v
Shared Feature Extractor
MobileNetV3-Large
        |
        +----------------------------------+
        |                                  |
        v                                  v
Emotion Head                         Feature Head
7 Emotion Classes                    8 Facial Attributes
        |                                  |
        v                                  v
Emotion Prediction                   Multi-Label Prediction
```

The architecture uses a shared visual backbone to learn general facial representations. Two task-specific prediction heads are then used:

* The emotion head predicts one of seven emotion classes.
* The feature head predicts eight facial attributes using multi-label classification.

## Multi-Task Learning Objective

The total training loss is defined as:

```math
L_{total} = L_{emotion} + \lambda L_{feature}
```

Where:

* `L_emotion` is the emotion classification loss.
* `L_feature` is the facial attribute prediction loss.
* `λ` controls the contribution of the facial attribute task.

The emotion classification task uses Focal Loss to reduce the impact of class imbalance. The facial attribute task uses Binary Cross-Entropy with logits.

## Occlusion-Aware Loss Masking

When a face mask is detected or synthetically applied, some mouth-related facial attributes become unobservable.

For samples where:

```text
mask = 1
```

the loss is masked for the following attributes:

* Smile
* Mouth Open
* Compressed Lips

This prevents the model from being penalized for predictions involving facial regions that are hidden by an occlusion.

```text
If the mouth is occluded,
the model should not be forced to predict mouth-related attributes.
```

## Project Structure

```text
XAI_FER/
│
├── data/                                  # Original facial expression datasets
│   ├── affectnet/                         # AffectNet dataset
│   ├── fer2013/                           # FER2013 dataset
│   └── raf_db/                            # RAF-DB dataset
│
├── data_preprocessing/                    # Data preprocessing pipeline
│   ├── csv/                               # Intermediate CSV files
│   ├── splits/                            # Dataset split files
│   ├── __init__.py
│   ├── all_features.csv                   # Extracted facial geometric features
│   ├── create_splits.py                   # Train, validation, and test split creation
│   ├── extract_features.py                # MediaPipe FaceMesh feature extraction
│   ├── generate_pseudo_labels.py          # Facial attribute pseudo-label generation
│   ├── merge_dataset.py                   # Dataset merging and label normalization
│   └── pseudo_labels.csv                  # Generated pseudo-labels
│
├── dataset_final/                         # Final processed dataset
│   ├── train/                             # Training images
│   ├── test/                              # Test images
│   └── metadata.csv                       # Dataset metadata and labels
│
├── diagnostics_report/                    # Evaluation reports and diagnostics
│   ├── misclassified_images/              # Incorrect prediction samples
│   ├── detailed_metrics_report.txt        # Detailed classification metrics
│   └── emotion_confusion_matrix.png       # Emotion confusion matrix
│
├── multitask/                             # Multi-task model implementation
│   ├── __init__.py
│   ├── emotion_head.py                    # Emotion classification head
│   ├── feature_head.py                    # Facial attribute prediction head
│   ├── model.py                           # Shared backbone definition
│   ├── multitask_model.py                 # Complete multi-task architecture
│   └── verify.py                          # Model architecture verification
│
├── phase0/                                # Feasibility study and data analysis
│   ├── step1_extract_features.py          # Landmark extraction benchmark
│   ├── step2_statistics.py                # Feature distribution analysis
│   ├── step3_feature_ranking.py           # Feature importance ranking
│   └── step4_generate_splits.py           # Initial dataset split generation
│
├── realtime/                              # Real-time webcam inference
│   └── webcam.py                          # Webcam emotion recognition application
│
├── robust_data/                           # Data loading and training utilities
│   ├── __init__.py
│   ├── dataset.py                         # Multi-task dataset loader
│   ├── loss.py                            # Focal Loss and masked feature loss
│   ├── sampler.py                         # Weighted random sampler
│   └── verify.py                          # Dataset and loader verification
│
├── training/                              # Training and inference scripts
│   ├── checkpoints/                       # Training checkpoints
│   ├── __init__.py
│   ├── metrics.py                         # Evaluation metrics
│   ├── optimize_threshold.py              # Per-feature threshold optimization
│   ├── predict.py                         # Single-image prediction
│   ├── train.py                           # Multi-task training pipeline
│   ├── utils.py                           # Shared training utilities
│   └── validate.py                        # Validation pipeline
│
├── xai/                                   # Explainable AI modules
│   ├── visualize_gradcam.py               # Grad-CAM visualization
│   └── xai_engine.py                      # XAI analysis and alignment scoring
│
├── checkpoints/                           # Best global model checkpoints
├── eval_test.py                           # Final test evaluation script
├── pyrefly.toml                           # Static type checking configuration
├── requirements.txt                       # Python dependencies
└── README.md
```

## Pipeline Overview

```text
Raw Datasets
    |
    v
Dataset Merging and Label Normalization
    |
    v
Feature Extraction with MediaPipe FaceMesh
    |
    v
Pseudo-Label Generation
    |
    v
Dataset Splitting
    |
    v
Multi-Task Training
    |
    v
Threshold Optimization
    |
    v
Evaluation and Diagnostics
    |
    v
Grad-CAM Explainability
    |
    v
Real-Time Webcam Inference
```

## Phase 0: Feasibility Study

The `phase0/` directory contains early experiments used to validate whether facial landmark geometry can support pseudo-label generation.

```text
phase0/
├── step1_extract_features.py
├── step2_statistics.py
├── step3_feature_ranking.py
└── step4_generate_splits.py
```

The feasibility study focuses on:

* Measuring MediaPipe FaceMesh detection reliability
* Analyzing landmark-derived facial features
* Identifying useful facial geometry ratios
* Ranking features based on their relationship with emotion labels
* Creating initial train, validation, and test splits

## Data Preprocessing

The preprocessing pipeline is implemented in the `data_preprocessing/` directory.

### Dataset Merging

The project combines FER2013, RAF-DB, and AffectNet into a unified dataset format.

```bash
python data_preprocessing/merge_dataset.py
```

The merged dataset normalizes emotion labels into the seven supported emotion classes.

### Facial Landmark Feature Extraction

MediaPipe FaceMesh is used to extract facial landmarks and calculate geometric facial features.

```bash
python data_preprocessing/extract_features.py
```

Examples of extracted features include:

* `mouth_aspect_ratio`
* `mouth_width_ratio`
* `eye_open_ratio`
* `eyebrow_distance_ratio`
* `eyebrow_eye_distance_ratio`
* `lip_compression_ratio`

The extracted features are stored in:

```text
data_preprocessing/all_features.csv
```

### Pseudo-Label Generation

Pseudo-labels are generated from facial geometry rules.

```bash
python data_preprocessing/generate_pseudo_labels.py
```

Example pseudo-label rules:

```text
High mouth aspect ratio
    -> mouth_open = 1

Large mouth width ratio
    -> smile = 1

Small eyebrow distance
    -> eyebrows_together = 1
```

Generated pseudo-labels are stored in:

```text
data_preprocessing/pseudo_labels.csv
```

### Dataset Splitting

Train and test splits are generated using:

```bash
python data_preprocessing/create_splits.py
```

The processed dataset is stored in:

```text
dataset_final/
├── train/
├── test/
└── metadata.csv
```

## Multi-Task Model

The multi-task architecture is implemented in the `multitask/` directory.

```text
multitask/
├── model.py
├── emotion_head.py
├── feature_head.py
└── multitask_model.py
```

### Shared Backbone

The model uses MobileNetV3-Large as the shared feature extractor. The backbone learns high-level facial representations from input face images.

### Emotion Head

The emotion head predicts one of seven emotion classes.

```text
Output shape: [batch_size, 7]
```

### Feature Head

The feature head predicts eight facial attributes using multi-label classification.

```text
Output shape: [batch_size, 8]
```

## Training

The training pipeline is located in the `training/` directory.

```bash
python training/train.py
```

During training, the system performs:

* Multi-task optimization
* Focal Loss for emotion classification
* Binary Cross-Entropy loss for facial attributes
* Occlusion-aware loss masking
* Weighted random sampling
* Validation after each epoch
* Best checkpoint saving

Best checkpoints are saved in:

```text
training/checkpoints/
```

## Threshold Optimization

Feature prediction thresholds are optimized individually using validation data.

```bash
python training/optimize_threshold.py
```

Instead of using a fixed threshold of `0.5`, each facial attribute receives an optimized threshold that maximizes its F1-score.

Example output:

```json
{
  "raised_eyebrows": 0.46,
  "lowered_eyebrows": 0.49,
  "eyebrows_together": 0.44,
  "eyes_wide_open": 0.35,
  "eyes_narrowed": 0.52,
  "mouth_open": 0.63,
  "smile": 0.42,
  "compressed_lips": 0.58
}
```

## Evaluation

Final test evaluation is performed using:

```bash
python eval_test.py
```

The evaluation process generates:

* Emotion Accuracy
* Emotion Macro F1-score
* Emotion Precision
* Emotion Recall
* Feature Macro F1-score
* Feature Precision
* Feature Recall
* Per-feature F1-score
* Emotion confusion matrix
* Misclassified image analysis

Evaluation outputs are stored in:

```text
diagnostics_report/
├── misclassified_images/
├── detailed_metrics_report.txt
└── emotion_confusion_matrix.png
```

## Explainable AI

The explainability module is located in the `xai/` directory.

```text
xai/
├── visualize_gradcam.py
└── xai_engine.py
```

Grad-CAM is used to visualize which facial regions contribute most to the predicted emotion.

```bash
python xai/visualize_gradcam.py
```

Expected attention regions include:

| Emotion  | Expected Facial Region |
| -------- | ---------------------- |
| Happy    | Mouth                  |
| Angry    | Eyebrows               |
| Surprise | Eyes and mouth         |
| Sad      | Eyes and mouth         |
| Fear     | Eyes and mouth         |

The XAI engine can compare Grad-CAM attention maps with predicted facial attributes to estimate explanation consistency.

## Real-Time Webcam Inference

The real-time webcam application is located in:

```text
realtime/webcam.py
```

Run the application with:

```bash
python realtime/webcam.py
```

Inference pipeline:

```text
Webcam Frame
    |
    v
Face Detection
    |
    v
Face Crop and Preprocessing
    |
    v
Multi-Task Model Inference
    |
    +----------------------+
    |                      |
    v                      v
Emotion Prediction     Feature Prediction
    |
    v
Live Visualization
```

Example output:

```text
Emotion: Happy
Confidence: 0.92

Smile: 0.95
Mouth Open: 0.74
Eyes Wide Open: 0.32
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

Typical dependencies include:

```text
torch
torchvision
opencv-python
mediapipe
albumentations
numpy
pandas
scikit-learn
matplotlib
tqdm
onnx
onnxruntime
psutil
```

## Recommended Evaluation Experiments

To evaluate the contribution of each component, compare the following configurations:

| Experiment                    | Emotion F1 | Feature F1 | Description                                      |
| ----------------------------- | ---------: | ---------: | ------------------------------------------------ |
| Baseline FER                  |          - |          - | Emotion-only classifier                          |
| Multi-task FER                |          - |          - | Emotion and facial attribute prediction          |
| Multi-task + Pseudo Labels    |          - |          - | Adds landmark-based facial attribute supervision |
| Multi-task + Weighted Sampler |          - |          - | Handles class imbalance                          |
| Multi-task + Loss Masking     |          - |          - | Ignores unobservable mouth features              |
| Full XAI-FER                  |          - |          - | Complete multi-task and explainability pipeline  |

## Future Improvements

* Add real-world datasets containing masks, glasses, and other facial occlusions
* Add an explicit validation split to `dataset_final/`
* Export trained models to ONNX for deployment
* Benchmark CPU inference speed and memory usage
* Add temporal smoothing for webcam predictions
* Add multi-face tracking
* Compare MobileNetV3-Large with MobileViT and EfficientNetV2
* Apply temperature scaling for probability calibration
* Add experiment tracking with MLflow or Weights & Biases
* Extend Grad-CAM validation with region overlap metrics
* Add Docker support for reproducible deployment

## License

This project is intended for academic and research purposes. Please ensure that the licenses of FER2013, RAF-DB, AffectNet, and other external datasets are respected before using or redistributing the data.
