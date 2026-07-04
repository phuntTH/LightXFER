import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def emotion_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def feature_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    return {"f1": f1_score(labels, preds, average="macro")}