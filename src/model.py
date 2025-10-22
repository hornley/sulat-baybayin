import torch
import torch.nn as nn
from torchvision import models
try:
    # torchvision >= 0.13
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


class BaybayinClassifier(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        # weights: None (no pretrained), or a ResNet18_Weights enum, or boolean True to request pretrained via old API
        # Use the new enum API only when an enum is provided. If a plain True is passed (older fallback), use pretrained=True.
        if ResNet18_Weights is not None and isinstance(weights, ResNet18_Weights):
            # modern torchvision: accept the weights enum
            self.backbone = models.resnet18(weights=weights)
        else:
            # handle older cases: if caller passed True, use pretrained=True (older API)
            if isinstance(weights, bool) and weights:
                try:
                    self.backbone = models.resnet18(pretrained=True)
                except Exception:
                    self.backbone = models.resnet18()
            else:
                # no pretrained weights
                self.backbone = models.resnet18()
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.backbone(x)
