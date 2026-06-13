from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED_TAG = "laion2b_s34b_b79k"
DEFAULT_TRANSFORMERS_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class ClipStatus:
    available: bool
    backend: str = "unavailable"
    model_name: str = ""
    pretrained_tag: str = ""
    pretrained: bool = False
    downloaded_or_cached: str = "unavailable"
    device: str = "cpu"
    downloads_allowed: bool = False
    backend_attempted: str = ""
    error_message: str = ""
    model: Any = None
    processor: Any = None
    preprocess: Any = None
    tokenizer: Any = None

    @property
    def message(self) -> str:
        return self.error_message


def _backend_order(preferred_backend: str) -> list[str]:
    backend = (preferred_backend or "open_clip").strip().lower()
    if backend == "auto":
        return ["open_clip", "transformers"]
    if backend in {"open_clip", "transformers"}:
        return [backend]
    return [backend]


@contextmanager
def _offline_weight_mode(enabled: bool):
    if not enabled:
        yield
        return
    keys = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]
    old_values = {key: os.environ.get(key) for key in keys}
    os.environ.update({key: "1" for key in keys})
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _open_clip_status(device: str, model_name: str, pretrained_tag: str, allow_download: bool, backend_attempted: str) -> ClipStatus:
    import open_clip  # type: ignore

    try:
        with _offline_weight_mode(not allow_download and not os.path.exists(pretrained_tag)):
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag, device=device)
    except Exception as exc:
        if not allow_download:
            raise RuntimeError(
                f"cached/local pretrained weights unavailable for {model_name}:{pretrained_tag}; "
                "set allow_pretrained_download true to permit weight download"
            ) from exc
        raise
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return ClipStatus(
        True,
        "open_clip",
        model_name,
        pretrained_tag,
        True,
        "download_allowed" if allow_download else "local_or_cached",
        device,
        allow_download,
        backend_attempted,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
    )


def _transformers_status(device: str, model_name: str, allow_download: bool, backend_attempted: str) -> ClipStatus:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore

    model = CLIPModel.from_pretrained(model_name, local_files_only=not allow_download).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=not allow_download)
    model.eval()
    return ClipStatus(
        True,
        "transformers",
        model_name,
        "",
        True,
        "download_allowed" if allow_download else "local_or_cached",
        device,
        allow_download,
        backend_attempted,
        model=model,
        processor=processor,
    )


def check_clip_available(
    device: str = "cpu",
    allow_download: bool = False,
    preferred_backend: str = "open_clip",
    model_name: str | None = None,
    pretrained_tag: str = DEFAULT_PRETRAINED_TAG,
) -> ClipStatus:
    backend_attempted = preferred_backend or "open_clip"
    errors: list[str] = []
    for backend in _backend_order(preferred_backend):
        if backend == "open_clip":
            try:
                return _open_clip_status(device, model_name or DEFAULT_MODEL_NAME, pretrained_tag, allow_download, backend_attempted)
            except Exception as exc:
                errors.append(f"open_clip unavailable: {exc}")
        elif backend == "transformers":
            try:
                return _transformers_status(device, model_name or DEFAULT_TRANSFORMERS_MODEL, allow_download, backend_attempted)
            except Exception as exc:
                errors.append(f"transformers CLIP unavailable: {exc}")
        else:
            errors.append(f"{backend} unavailable: unsupported CLIP backend")

    return ClipStatus(
        False,
        "unavailable",
        model_name or "",
        pretrained_tag if "open_clip" in _backend_order(preferred_backend) else "",
        False,
        "unavailable",
        device,
        allow_download,
        backend_attempted,
        "; ".join(errors) + ". Install optional CLIP support with `pip install open_clip_torch` or `pip install transformers`.",
    )


def load_clip_model(
    device: str = "cpu",
    allow_download: bool = False,
    preferred_backend: str = "open_clip",
    model_name: str | None = None,
    pretrained_tag: str = DEFAULT_PRETRAINED_TAG,
) -> ClipStatus:
    return check_clip_available(
        device=device,
        allow_download=allow_download,
        preferred_backend=preferred_backend,
        model_name=model_name,
        pretrained_tag=pretrained_tag,
    )


def _to_pil_list(images: torch.Tensor):
    from PIL import Image  # type: ignore

    imgs = images.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return [Image.fromarray((img * 255).astype(np.uint8)) for img in imgs]


def encode_images(status: ClipStatus, images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    if not status.available:
        raise RuntimeError(status.error_message or "CLIP is unavailable")
    if status.backend == "transformers":
        inputs = status.processor(images=_to_pil_list(images), return_tensors="pt").to(device)
        with torch.no_grad():
            feats = status.model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    processed = torch.stack([status.preprocess(img) for img in _to_pil_list(images)]).to(device)
    with torch.no_grad():
        feats = status.model.encode_image(processed)
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def encode_text_prompts(status: ClipStatus, prompts: list[str], device: str = "cpu") -> torch.Tensor:
    if not status.available:
        raise RuntimeError(status.error_message or "CLIP is unavailable")
    if status.backend == "transformers":
        inputs = status.processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = status.model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tokens = status.tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = status.model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class ClipZeroShotClassifier:
    def __init__(self, status: ClipStatus, class_names: list[str], prompts: list[str] | None = None, device: str = "cpu") -> None:
        self.status = status
        self.class_names = class_names
        self.prompts = prompts or [f"a simple image of a {name}" for name in class_names]
        self.device = device
        self.model_name = status.model_name
        self.pretrained = bool(status.pretrained)
        self.zero_shot = True
        self.linear_probe = False
        self.warning = ""
        self._text_features: torch.Tensor | None = None

    def predict(self, images: torch.Tensor) -> dict[str, Any]:
        result = predict_zero_shot(images, self.class_names, self.status, self.device, self.prompts)
        if not result.get("available", False):
            raise RuntimeError(result.get("message", "CLIP unavailable"))
        return result


def predict_zero_shot(
    images: torch.Tensor,
    class_names: list[str],
    status: ClipStatus | None = None,
    device: str = "cpu",
    prompts: list[str] | None = None,
) -> dict[str, Any]:
    status = status or load_clip_model(device)
    if not status.available:
        return {"available": False, "message": status.error_message, "backend": status.backend}
    prompts = prompts or [f"a simple image of a {name}" for name in class_names]
    image_features = encode_images(status, images, device)
    text_features = encode_text_prompts(status, prompts, device)
    logits = 100.0 * image_features @ text_features.T
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return {
        "available": True,
        "backend": status.backend,
        "logits": logits.detach().cpu(),
        "probabilities": probs.detach().cpu(),
        "predictions": pred.detach().cpu(),
        "confidence": conf.detach().cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--backend", choices=["open_clip", "transformers", "auto"], default="open_clip")
    parser.add_argument("--model-name")
    parser.add_argument("--pretrained-tag", default=DEFAULT_PRETRAINED_TAG)
    args = parser.parse_args()
    default_model = DEFAULT_TRANSFORMERS_MODEL if args.backend == "transformers" else DEFAULT_MODEL_NAME
    status = check_clip_available(
        allow_download=args.allow_download,
        preferred_backend=args.backend,
        model_name=args.model_name or default_model,
        pretrained_tag=args.pretrained_tag,
    )
    print(f"CLIP available: {'yes' if status.available else 'no'}")
    print(f"Backend attempted: {status.backend_attempted}")
    print(f"Backend used: {status.backend}")
    print(f"Model name: {status.model_name or 'unavailable'}")
    print(f"Pretrained tag: {status.pretrained_tag or 'none'}")
    print(f"Pretrained weights loaded: {'yes' if status.pretrained else 'no'}")
    print(f"Device: {status.device}")
    if not status.available:
        print(f"Error: {status.error_message}")
        print("Install help: pip install open_clip_torch  # or: pip install transformers")


if __name__ == "__main__":
    main()
