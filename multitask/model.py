import torch.nn as nn
import timm

class MobileNetV3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            num_classes=0
        )

    def forward(self, x):
        return self.backbone(x)