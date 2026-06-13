from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from causal_reliability.real_models.clip_zero_shot import ClipStatus, load_clip_model, predict_zero_shot
from causal_reliability.real_models.pretrained_classifier import ClassifierStatus, make_small_cnn, train_classifier, try_torchvision_classifier


@dataclass
class RealModel:
    mode: str
    class_names: list[str]
    status: ClipStatus | ClassifierStatus
    device: str = "cpu"

    @property
    def model_name(self) -> str:
        if self.mode == "clip":
            return f"CLIP ({self.status.backend})"
        return self.status.model_name

    @property
    def pretrained(self) -> bool:
        return bool(self.mode == "clip" or getattr(self.status, "pretrained", False))

    @property
    def zero_shot(self) -> bool:
        return self.mode == "clip"

    @property
    def linear_probe(self) -> bool:
        return bool(getattr(self.status, "linear_probe", False))

    @property
    def warning(self) -> str:
        return str(getattr(self.status, "message", "") or getattr(self.status, "warning", ""))

    def predict(self, images: torch.Tensor) -> dict[str, Any]:
        images = images.to(self.device)
        if self.mode == "clip":
            return predict_zero_shot(images, self.class_names, self.status, self.device)
        model = self.status.model
        model.eval()
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confidence, predictions = probs.max(dim=1)
        return {
            "available": True,
            "backend": self.status.model_name,
            "logits": logits.detach().cpu(),
            "probabilities": probs.detach().cpu(),
            "predictions": predictions.detach().cpu(),
            "confidence": confidence.detach().cpu(),
        }


def load_real_model(
    class_names: list[str],
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    cfg: dict[str, Any] | None = None,
    device: str = "cpu",
) -> RealModel:
    cfg = cfg or {}
    preference = str(cfg.get("mode", "auto"))
    if preference in {"auto", "clip"}:
        clip_status = load_clip_model(device, allow_download=bool(cfg.get("allow_weight_download", False)))
        if clip_status.available:
            return RealModel("clip", class_names, clip_status, device)
        if preference == "clip" and not cfg.get("fallback_on_clip_unavailable", True):
            return RealModel("clip", class_names, clip_status, device)
    if preference in {"auto", "torchvision"}:
        status = try_torchvision_classifier(
            len(class_names),
            device=device,
            allow_untrained=bool(cfg.get("allow_untrained_torchvision", False)),
            allow_download=bool(cfg.get("allow_weight_download", False)),
        )
        if status is not None:
            status.model = train_classifier(
                status.model,
                train_images.to(device),
                train_labels.to(device),
                epochs=int(cfg.get("probe_epochs", 3)),
                lr=float(cfg.get("probe_lr", 0.01)),
                batch_size=int(cfg.get("batch_size", 32)),
            )
            return RealModel("classifier", class_names, status, device)
    status = make_small_cnn(len(class_names), device=device)
    status.model = train_classifier(
        status.model,
        train_images.to(device),
        train_labels.to(device),
        epochs=int(cfg.get("fallback_epochs", 12)),
        lr=float(cfg.get("fallback_lr", 0.01)),
        batch_size=int(cfg.get("batch_size", 32)),
    )
    return RealModel("classifier", class_names, status, device)
