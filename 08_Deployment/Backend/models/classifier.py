"""
BirdSense Backend — BirdClassifier Model
EfficientNet-B2 bird species classifier (ported from Streamlit version).
"""

import torch
import torch.nn as nn
import timm


class BirdClassifier(nn.Module):
    """EfficientNet-B2-based bird species classifier."""

    def __init__(self, num_classes: int = 87, dropout_rate: float = 0.4):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            "tf_efficientnet_b2_ns",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        num_features = self.backbone.num_features  # 1408 for B2

        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (for explainability)."""
        return self.backbone(x)
