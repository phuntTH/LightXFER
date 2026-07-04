import torch.nn as nn

from .emotion_head import EmotionHead
from .feature_head import FeatureHead
from .model import MobileNetV3Backbone


class MultiTaskFER(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV3Backbone()
        self.emotion_head = EmotionHead()
        self.feature_head = FeatureHead()

    def forward(self, x, return_features=False):
        shared_feature = self.backbone(x)

        emotion_logits = self.emotion_head(shared_feature)
        feature_logits = self.feature_head(shared_feature)

        outputs = {
            "emotion_logits": emotion_logits,
            "feature_logits": feature_logits,
        }

        if return_features:
            outputs["shared_feature"] = shared_feature

        return outputs

    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, num_blocks=3):
        self.backbone.train()

        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone.backbone, "conv_head"):
            for param in self.backbone.backbone.conv_head.parameters():
                param.requires_grad = True

        if hasattr(self.backbone.backbone, "blocks"):
            for block in self.backbone.backbone.blocks[-num_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True