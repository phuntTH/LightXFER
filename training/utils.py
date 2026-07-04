import numpy as np
import torch


class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping Counter {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def save_checkpoint(
    model, optimizer, epoch, emotion_f1, feature_f1, save_path
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "emotion_f1": emotion_f1,
            "feature_f1": feature_f1,
        },
        save_path,
    )


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def print_epoch_result(epoch, train_loss, emotion_f1, feature_f1):
    print("\n" + "=" * 60)
    print(f"Epoch {epoch}")
    print(f"Train Loss   : {train_loss:.4f}")
    print(f"Emotion F1   : {emotion_f1:.4f}")
    print(f"Feature F1   : {feature_f1:.4f}")
    print("=" * 60)