import torch

from multitask_model import MultiTaskFER

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

model = (MultiTaskFER().to(DEVICE))

dummy = torch.randn(8, 3, 224, 224).to(DEVICE)

outputs = model(dummy)

print("\nBackbone")


print("\nEmotion")

print(outputs["emotion_logits"].shape)

print("\nFeature")

print(outputs["feature_logits"].shape)