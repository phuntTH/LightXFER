import numpy as np
import torch
from metrics import emotion_metrics, feature_metrics


def validate(model, loader, device):
    model.eval()

    emotion_preds = []
    emotion_labels = []

    feature_preds = []
    feature_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            emotions = batch["emotion"]
            features = batch["features"]

            outputs = model(images)
            emotion_logits = outputs["emotion_logits"]
            feature_logits = outputs["feature_logits"]

            pred_emotion = torch.argmax(emotion_logits, dim=1)
            pred_feature = torch.sigmoid(feature_logits) > 0.5

            emotion_preds.extend(pred_emotion.cpu().numpy())
            emotion_labels.extend(emotions.numpy())

            feature_preds.extend(pred_feature.cpu().numpy())
            feature_labels.extend(features.numpy())

    emotion_result = emotion_metrics(emotion_preds, emotion_labels)
    feature_result = feature_metrics(feature_preds, feature_labels)

    return emotion_result, feature_result