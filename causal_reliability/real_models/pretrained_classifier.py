from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ClassifierStatus:
    model: nn.Module
    model_name: str
    pretrained: bool
    linear_probe: bool
    backbone_frozen: bool
    warning: str = ""


def try_torchvision_classifier(n_classes: int, device: str = "cpu", allow_untrained: bool = True, allow_download: bool = False) -> ClassifierStatus | None:
    try:
        from torchvision import models  # type: ignore
    except Exception:
        return None
    attempts: list[tuple[str, Any, str]] = [
        ("resnet18", models.resnet18, "ResNet18_Weights"),
        ("efficientnet_b0", models.efficientnet_b0, "EfficientNet_B0_Weights"),
        ("vit_b_16", models.vit_b_16, "ViT_B_16_Weights"),
    ]
    last_error = ""
    for name, builder, weights_name in attempts:
        try:
            if not allow_download:
                raise RuntimeError("pretrained torchvision weight downloads are disabled")
            weights_enum = getattr(models, weights_name)
            model = builder(weights=weights_enum.DEFAULT)
            pretrained = True
        except Exception as exc:
            last_error = f"{name} pretrained weights unavailable: {exc}"
            if not allow_untrained:
                continue
            try:
                model = builder(weights=None)
                pretrained = False
            except Exception as inner:
                last_error = f"{last_error}; untrained architecture failed: {inner}"
                continue
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, n_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)
        elif hasattr(model, "heads") and hasattr(model.heads, "head"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, n_classes)
        else:
            continue
        return ClassifierStatus(
            model=model.to(device),
            model_name=f"torchvision_{name}",
            pretrained=pretrained,
            linear_probe=True,
            backbone_frozen=True,
            warning="" if pretrained else f"Non-pretrained torchvision fallback used. {last_error}",
        )
    return None


def make_small_cnn(n_classes: int, device: str = "cpu") -> ClassifierStatus:
    return ClassifierStatus(
        model=SmallCNN(n_classes).to(device),
        model_name="local_small_cnn",
        pretrained=False,
        linear_probe=False,
        backbone_frozen=False,
        warning="Non-pretrained fallback: local small CNN trained only on the controlled ID shortcut data.",
    )


def train_classifier(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int = 10, lr: float = 0.01, batch_size: int = 32) -> nn.Module:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    n = len(x)
    for _ in range(max(1, epochs)):
        order = torch.randperm(n, device=x.device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            logits = model(x[idx])
            loss = nn.functional.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model
