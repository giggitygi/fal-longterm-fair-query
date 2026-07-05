from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.projector(self.encoder(x))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def make_model(dataset_name: str, num_classes: int) -> nn.Module:
    in_channels = 1 if dataset_name in {"MNIST", "FashionMNIST"} else 3
    return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
