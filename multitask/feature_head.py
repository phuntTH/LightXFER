import torch.nn as nn

class FeatureHead(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),             
            nn.Dropout(0.3),
            nn.Linear(512, 8)
        )

    def forward(self, x):
        return self.head(x)