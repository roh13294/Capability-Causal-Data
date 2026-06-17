from __future__ import annotations

"""Real-OpenCLIP CSA adapter/LoRA pilot.

This module moves the Counterfactual Stability Alignment (CSA) pilot beyond the
frozen-embedding linear-head proof-of-concept (see
:mod:`causal_reliability.training.csa_trainer`, which it deliberately does **not**
modify) by adapting a small part of a *real* OpenCLIP model and measuring whether
counterfactual stability improves without destroying clean accuracy or zero-shot
behaviour.

Bounded framing
---------------
This is a *bounded CSA-LoRA pilot*. The only claim is whether a CIC-style
stability regularizer can reduce measured shortcut reliance in an **adapted**
CLIP model under held-out finite-candidate interventions. It is **not** an
RLHF/DPO replacement, **not** universal robustness, **not** open-world shortcut
discovery, **not** deployment validation, and **not** a replacement for the
finalized STS report.

What gets trained
-----------------
Most of OpenCLIP stays frozen. We only train the *smallest feasible* trainable
module:

* If PEFT/LoRA tooling is available it would be used to inject LoRA into the
  vision tower. PEFT is an optional dependency; when it is unavailable this
  pilot **skips LoRA cleanly** and instead trains a minimal residual
  **adapter/head over the frozen CLIP image embeddings**, which is labelled
  exactly as that (``adapter_head_only`` — *not* LoRA).
* The frozen zero-shot text head (CLIP text-prompt features) is the classifier,
  so the adapted model is still a zero-shot/prompt model and zero-shot collapse
  is directly measurable.

The adapter is initialised to the identity (its up-projection starts at zero) so
an untrained adapter reproduces the frozen zero-shot model exactly.

Data
----
Controlled text-overlay shortcut stimuli rendered as real images and encoded with
real CLIP:

* ``train``        — overlay text correlated with the object label (the shortcut).
* ``val``          — in-distribution validation, random overlay.
* ``held_out_val`` — **held-out text overlays** (a different overlay rendering
  style): did the adapter merely memorize the training overlay appearance?
* ``transfer_val`` — an optional **semantic-decoy icon** family (no text): does
  the learned stability transfer to a different, non-text shortcut family?

For every example we materialise the ``observed`` image (overlay actually
painted), the ``clean`` image (overlay removed), a ``misleading`` image (a
competing-class overlay), and the finite ``candidates`` bank (one image per
candidate overlay word) used for the CIC penalty.

Fallback
--------
If real OpenCLIP weights cannot be loaded (no cache / downloads disabled) the
pilot falls back to the controlled synthetic frozen embeddings used by the linear
head pilot and records why, so the run never crashes and never silently claims a
real-model result.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_reliability.training.csa_loss import (
    cic_instability_penalty,
    csa_total_loss,
)
from causal_reliability.training.csa_trainer import (
    DataConfig as SyntheticDataConfig,
    EmbeddingSplit,
    build_synthetic_dataset,
    clean_class_means,
)


TRAINING_MODES = ("frozen", "plain_ft", "cf_aug", "csa")

# Shape object classes (the causal content) for the text-overlay family.
SHAPE_CLASSES = ["circle", "square", "triangle", "star"]
# Semantic-decoy icon classes (the transfer family — no written words anywhere).
DECOY_CLASSES = ["sun", "heart", "leaf", "moon"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class LoraDataConfig:
    n_train: int = 256
    n_val: int = 128
    n_held_out_val: int = 128
    n_transfer_val: int = 128
    shortcut_corr: float = 0.9
    image_size: int = 224
    enable_transfer_family: bool = True
    prompt_template: str = "a photo of a {label}"
    decoy_prompt_template: str = "a simple colorful icon of a {label}"


@dataclass
class LoraTrainConfig:
    epochs: int = 3
    lr: float = 0.01
    batch_size: int = 32
    weight_decay: float = 0.0


@dataclass
class AdapterConfig:
    bottleneck: int = 32
    alpha: float = 1.0
    # LoRA rank requested if PEFT/LoRA tooling is available; ignored otherwise.
    lora_rank: int = 4
    prefer_lora: bool = True


@dataclass
class LoraCsaConfig:
    lambda_stability: float = 1.0
    lambda_cic: float = 1.0
    lambda_preservation: float = 0.5
    preservation_mode: str = "kl"


@dataclass
class ManualLoraConfig:
    """Configuration for the ``manual_lora_visual`` mode (actual LoRA on the
    OpenCLIP visual tower via the internal PyTorch wrapper — no PEFT)."""

    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_last_blocks: int = 2
    target_modules: tuple = ("c_fc", "c_proj", "out_proj")
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 8
    encode_batch_size: int = 16
    train_text: bool = False  # keep prompt/text embeddings fixed unless enabled

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "ManualLoraConfig":
        raw = dict(raw or {})
        tm = raw.get("target_modules")
        target_modules = tuple(tm) if tm else ("c_fc", "c_proj", "out_proj")
        return cls(
            rank=int(raw.get("rank", 4)),
            alpha=float(raw.get("alpha", 8.0)),
            dropout=float(raw.get("dropout", 0.0)),
            target_last_blocks=int(raw.get("target_last_blocks", 2)),
            target_modules=target_modules,
            lr=float(raw.get("lr", 2e-4)),
            epochs=int(raw.get("epochs", 1)),
            batch_size=int(raw.get("batch_size", 8)),
            encode_batch_size=int(raw.get("encode_batch_size", 16)),
            train_text=bool(raw.get("train_text", False)),
        )


@dataclass
class ModelConfig:
    backend: str = "open_clip"
    model_name: str = "ViT-B-32"
    pretrained_tag: str = "laion2b_s34b_b79k"
    transformers_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    allow_download: bool = False
    logit_scale: float = 100.0
    encode_batch_size: int = 32


@dataclass
class LoraPilotConfig:
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: LoraDataConfig = field(default_factory=LoraDataConfig)
    train: LoraTrainConfig = field(default_factory=LoraTrainConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    csa: LoraCsaConfig = field(default_factory=LoraCsaConfig)
    manual_lora: ManualLoraConfig = field(default_factory=ManualLoraConfig)
    cache_dir: str = "results/csa_lora_pilot/cache"
    use_embedding_cache: bool = True
    # Go / no-go thresholds (pre-registered) for the adapter_head_only mode.
    min_instability_drop_rel: float = 0.20
    max_clean_accuracy_drop: float = 0.03
    min_held_out_overlay_gain: float = 0.05
    max_transfer_regression: float = 0.03
    strong_transfer_gain: float = 0.05
    # Stronger go / no-go thresholds (pre-registered) for manual_lora_visual.
    ml_min_instability_drop_rel: float = 0.20
    ml_max_clean_drop_vs_frozen: float = 0.03
    ml_min_held_out_gain_vs_plain: float = 0.05
    ml_max_held_out_regression_vs_cf: float = 0.02
    ml_max_transfer_regression_vs_plain: float = 0.03
    ml_strong_cf_gain: float = 0.05

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "LoraPilotConfig":
        raw = dict(raw or {})
        gng = dict(raw.get("go_no_go", {}) or {})
        mlg = dict(raw.get("manual_lora_go_no_go", {}) or {})
        return cls(
            seed=int(raw.get("seed", 0)),
            model=ModelConfig(**(raw.get("model", {}) or {})),
            data=LoraDataConfig(**(raw.get("data", {}) or {})),
            train=LoraTrainConfig(**(raw.get("train", {}) or {})),
            adapter=AdapterConfig(**(raw.get("adapter", {}) or {})),
            csa=LoraCsaConfig(**(raw.get("csa", {}) or {})),
            manual_lora=ManualLoraConfig.from_raw(raw.get("manual_lora", {})),
            cache_dir=str(raw.get("cache_dir", "results/csa_lora_pilot/cache")),
            use_embedding_cache=bool(raw.get("use_embedding_cache", True)),
            min_instability_drop_rel=float(gng.get("min_instability_drop_rel", 0.20)),
            max_clean_accuracy_drop=float(gng.get("max_clean_accuracy_drop", 0.03)),
            min_held_out_overlay_gain=float(gng.get("min_held_out_overlay_gain", 0.05)),
            max_transfer_regression=float(gng.get("max_transfer_regression", 0.03)),
            strong_transfer_gain=float(gng.get("strong_transfer_gain", 0.05)),
            ml_min_instability_drop_rel=float(mlg.get("min_instability_drop_rel", 0.20)),
            ml_max_clean_drop_vs_frozen=float(mlg.get("max_clean_drop_vs_frozen", 0.03)),
            ml_min_held_out_gain_vs_plain=float(mlg.get("min_held_out_gain_vs_plain", 0.05)),
            ml_max_held_out_regression_vs_cf=float(mlg.get("max_held_out_regression_vs_cf", 0.02)),
            ml_max_transfer_regression_vs_plain=float(mlg.get("max_transfer_regression_vs_plain", 0.03)),
            ml_strong_cf_gain=float(mlg.get("strong_cf_gain", 0.05)),
        )


# --------------------------------------------------------------------------- #
# Trainable module: minimal residual adapter over frozen embeddings
# --------------------------------------------------------------------------- #
def lora_dependencies_available() -> bool:
    """True only if real LoRA tooling (PEFT) can be imported.

    When this is ``False`` the pilot does not fabricate a LoRA result; it trains
    a labelled adapter/head instead.
    """

    try:
        import peft  # noqa: F401

        return True
    except Exception:
        return False


class EmbeddingAdapter(nn.Module):
    """Residual bottleneck adapter applied to frozen CLIP image embeddings.

    ``z' = z + alpha * W_up(relu(W_down(z)))``. ``W_up`` is initialised to zero
    so an untrained adapter is the identity map and reproduces the frozen
    zero-shot model exactly. This is a *head/adapter-only* module over frozen
    embeddings — it is **not** LoRA on the backbone weights.
    """

    def __init__(self, dim: int, bottleneck: int = 32, alpha: float = 1.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.alpha = float(alpha)
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(F.relu(self.down(x)))


def resolve_trainable_module(cfg: LoraPilotConfig) -> dict[str, Any]:
    """Decide and *describe* the trainable module without constructing it.

    Returns a record with the exact module type used and whether LoRA was used or
    skipped. PEFT is optional; if it is unavailable we skip LoRA cleanly and use
    a labelled adapter/head instead.
    """

    lora_ok = lora_dependencies_available()
    use_lora = bool(cfg.adapter.prefer_lora and lora_ok)
    if use_lora:
        return {
            "lora_available": True,
            "lora_used": True,
            "adapter_used": False,
            "trainable_module_type": (
                f"PEFT LoRA (rank={cfg.adapter.lora_rank}) on the OpenCLIP vision tower"
            ),
            "note": "PEFT/LoRA tooling available; LoRA injected into the vision tower.",
        }
    return {
        "lora_available": lora_ok,
        "lora_used": False,
        "adapter_used": True,
        "trainable_module_type": (
            "adapter_head_only: residual bottleneck adapter over FROZEN CLIP image "
            "embeddings (NOT LoRA)"
        ),
        "note": (
            "PEFT/LoRA tooling unavailable; LoRA skipped cleanly. A minimal "
            "trainable adapter/head over frozen image embeddings is used instead "
            "and is labelled adapter/head-only, not LoRA."
            if not lora_ok
            else "LoRA disabled by config; using adapter/head-only module."
        ),
    }


# --------------------------------------------------------------------------- #
# Frozen classifier (the head the adapter feeds into)
# --------------------------------------------------------------------------- #
@dataclass
class FrozenClassifier:
    """A fixed (non-trainable) classifier mapping an embedding to class logits.

    ``kind="clip_text"`` uses cosine similarity to frozen text-prompt features
    (the zero-shot head). ``kind="ncm"`` uses negative squared distance to clean
    class means (the synthetic-fallback reader).
    """

    kind: str
    logit_scale: float = 100.0
    text_features: Optional[torch.Tensor] = None  # [K, D]
    class_means: Optional[torch.Tensor] = None  # [K, D]

    def logits(self, emb: torch.Tensor) -> torch.Tensor:
        flat = emb.reshape(-1, emb.shape[-1])
        if self.kind == "clip_text":
            z = F.normalize(flat, dim=-1)
            out = self.logit_scale * (z @ self.text_features.T)
        elif self.kind == "ncm":
            diff = flat.unsqueeze(1) - self.class_means
            out = -(diff ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown classifier kind: {self.kind!r}")
        return out.reshape(*emb.shape[:-1], out.shape[-1])


def _apply_adapter(adapter: Optional[nn.Module], emb: torch.Tensor) -> torch.Tensor:
    if adapter is None:
        return emb
    flat = emb.reshape(-1, emb.shape[-1])
    out = adapter(flat)
    return out.reshape(*emb.shape[:-1], out.shape[-1])


def model_logits(
    adapter: Optional[nn.Module], classifier: FrozenClassifier, emb: torch.Tensor
) -> torch.Tensor:
    return classifier.logits(_apply_adapter(adapter, emb))


# --------------------------------------------------------------------------- #
# Dataset (embeddings + frozen classifier(s))
# --------------------------------------------------------------------------- #
@dataclass
class LoraCsaDataset:
    train: EmbeddingSplit
    val: EmbeddingSplit
    held_out_val: EmbeddingSplit
    transfer_val: Optional[EmbeddingSplit]
    classifier: FrozenClassifier
    transfer_classifier: Optional[FrozenClassifier]
    embed_dim: int
    num_classes: int
    embedding_source: str
    backend: str
    clip_loaded: bool
    source_note: str = ""


# ---- image rendering for the text-overlay family --------------------------- #
def _shape_points(label: int, size: int):
    import numpy as np

    cx, cy, r = size * 0.5, size * 0.48, size * 0.22
    if label == 2:  # triangle
        return [(cx, cy - 1.15 * r), (cx - 1.15 * r, cy + r), (cx + 1.15 * r, cy + r)]
    points = []
    for k in range(10):  # star
        rad = r if k % 2 == 0 else r * 0.45
        ang = -np.pi / 2 + k * np.pi / 5
        points.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return points


# Two overlay rendering *styles*. ``train`` is the style the adapter is trained
# on; ``held_out`` is a visually different overlay style (colour + position +
# font) used to build the held-out-text-overlay split.
_OVERLAY_STYLES = {
    "train": {"text_color": (180, 20, 24), "box_color": (255, 255, 255), "y_frac": 0.72, "font_div": 7},
    "held_out": {"text_color": (20, 40, 170), "box_color": (250, 244, 210), "y_frac": 0.12, "font_div": 6},
}


def _render_shape_overlay(label: int, overlay_word: str, size: int, style: str, draw_overlay: bool):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    st = _OVERLAY_STYLES[style]
    img = Image.new("RGB", (size, size), (238, 240, 235))
    draw = ImageDraw.Draw(img)
    box = [size * 0.28, size * 0.22, size * 0.72, size * 0.66]
    fill = (32, 34, 36)
    if label == 0:
        draw.ellipse(box, fill=fill)
    elif label == 1:
        draw.rectangle(box, fill=fill)
    else:
        draw.polygon(_shape_points(label, size), fill=fill)

    if draw_overlay and overlay_word:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(10, size // st["font_div"]))
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), overlay_word, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(4, (size - tw) // 2)
        y = int(size * st["y_frac"])
        pad = max(4, size // 45)
        draw.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], radius=3, fill=st["box_color"])
        draw.text((x, y), overlay_word, font=font, fill=st["text_color"])

    return np.asarray(img).astype(np.float32) / 255.0


def _render_decoy(label: int, decoy_label: Optional[int], size: int):
    """Render a centred causal icon plus an optional competing-class corner icon."""

    from causal_reliability.data.clip_semantic_decoy_shortcuts import (
        render_semantic_decoy_image,
    )

    corner = 3  # fixed bottom-right corner keeps the candidate bank comparable.
    arr, _ = render_semantic_decoy_image(
        label, decoy_label, corner, size=size, draw_decoy=decoy_label is not None
    )
    return arr


def _images_to_tensor(images):
    import numpy as np

    arr = np.stack(images).astype(np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


# ---- split construction (image-space plans, then a single CLIP encode) ----- #
def _overlay_plan(n: int, num_classes: int, shortcut_corr: float, correlated: bool, gen: torch.Generator):
    y = torch.randint(0, num_classes, (n,), generator=gen)
    if correlated:
        agree = torch.rand(n, generator=gen) < shortcut_corr
        rand_overlay = torch.randint(0, num_classes, (n,), generator=gen)
        overlay = torch.where(agree, y, rand_overlay)
    else:
        overlay = torch.randint(0, num_classes, (n,), generator=gen)
    decoy = (y + 1) % num_classes  # deterministic competing/misleading overlay
    return y, overlay, decoy


def _encode_split_clip(
    encode_fn,
    *,
    y: torch.Tensor,
    overlay: torch.Tensor,
    decoy: torch.Tensor,
    render_obs,
    render_clean,
    num_classes: int,
) -> EmbeddingSplit:
    """Render every required image for a split and encode them with CLIP.

    ``render_obs(label, word_label)`` renders an image carrying the overlay of
    ``word_label`` on object ``label``. ``render_clean(label)`` renders the
    overlay-free image. Candidates stack one image per candidate overlay word.
    """

    n = int(y.shape[0])
    images = []
    # observed (n), clean (n), misleading (n), candidates (n * num_classes)
    for i in range(n):
        images.append(render_obs(int(y[i]), int(overlay[i])))
    for i in range(n):
        images.append(render_clean(int(y[i])))
    for i in range(n):
        images.append(render_obs(int(y[i]), int(decoy[i])))
    for i in range(n):
        for c in range(num_classes):
            images.append(render_obs(int(y[i]), c))

    feats = encode_fn(_images_to_tensor(images))  # [total, D]
    observed = feats[:n]
    clean = feats[n : 2 * n]
    misleading = feats[2 * n : 3 * n]
    candidates = feats[3 * n :].reshape(n, num_classes, -1)
    return EmbeddingSplit(observed, clean, misleading, candidates, y)


def _config_hash(cfg: LoraPilotConfig, status_tag: str) -> str:
    payload = {
        "seed": cfg.seed,
        "data": cfg.data.__dict__,
        "model": {k: getattr(cfg.model, k) for k in ("model_name", "pretrained_tag", "backend", "logit_scale")},
        "status_tag": status_tag,
        "version": 2,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def build_clip_dataset(cfg: LoraPilotConfig) -> LoraCsaDataset:
    """Build the CLIP-embedding dataset, falling back to synthetic if needed."""

    from causal_reliability.real_models.clip_zero_shot import (
        check_clip_available,
        encode_images,
        encode_text_prompts,
    )

    status = check_clip_available(
        device=cfg.model.device,
        allow_download=cfg.model.allow_download,
        preferred_backend=cfg.model.backend,
        model_name=cfg.model.model_name,
        pretrained_tag=cfg.model.pretrained_tag,
        transformers_model_name=cfg.model.transformers_model_name,
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        note = (
            "real OpenCLIP weights unavailable "
            f"({status.error_message or 'not loaded'}); fell back to controlled "
            "synthetic frozen embeddings"
        )
        return build_synthetic_fallback(cfg, note)

    status_tag = f"{status.backend}:{status.model_name}:{status.pretrained_tag}"
    cache_path = Path(cfg.cache_dir) / f"clip_embeddings_{_config_hash(cfg, status_tag)}.pt"
    if cfg.use_embedding_cache and cache_path.exists():
        try:
            return _load_cached_dataset(cache_path, status, status_tag)
        except Exception:
            pass  # fall through and recompute

    device = cfg.model.device

    def encode_fn(images: torch.Tensor) -> torch.Tensor:
        bs = max(1, cfg.model.encode_batch_size)
        out = []
        for i in range(0, len(images), bs):
            out.append(encode_images(status, images[i : i + bs], device).detach().cpu())
        return torch.cat(out, dim=0)

    gen = torch.Generator().manual_seed(cfg.seed)
    k = len(SHAPE_CLASSES)
    sz = cfg.data.image_size

    def render_obs_style(style):
        return lambda label, word: _render_shape_overlay(label, SHAPE_CLASSES[word], sz, style, True)

    def render_clean(label):
        return _render_shape_overlay(label, "", sz, "train", False)

    # train: overlay correlated with label, train style.
    y, ov, dc = _overlay_plan(cfg.data.n_train, k, cfg.data.shortcut_corr, True, gen)
    train = _encode_split_clip(encode_fn, y=y, overlay=ov, decoy=dc, render_obs=render_obs_style("train"), render_clean=render_clean, num_classes=k)
    # val: random overlay, train style.
    y, ov, dc = _overlay_plan(cfg.data.n_val, k, cfg.data.shortcut_corr, False, gen)
    val = _encode_split_clip(encode_fn, y=y, overlay=ov, decoy=dc, render_obs=render_obs_style("train"), render_clean=render_clean, num_classes=k)
    # held_out_val: random overlay, HELD-OUT overlay style.
    y, ov, dc = _overlay_plan(cfg.data.n_held_out_val, k, cfg.data.shortcut_corr, False, gen)
    held_out = _encode_split_clip(encode_fn, y=y, overlay=ov, decoy=dc, render_obs=render_obs_style("held_out"), render_clean=render_clean, num_classes=k)

    # Frozen zero-shot text head for the shape family.
    prompts = [cfg.data.prompt_template.format(label=name) for name in SHAPE_CLASSES]
    text_feats = encode_text_prompts(status, prompts, device).detach().cpu()
    classifier = FrozenClassifier("clip_text", cfg.model.logit_scale, text_features=text_feats)

    transfer = None
    transfer_classifier = None
    if cfg.data.enable_transfer_family:
        kd = len(DECOY_CLASSES)
        y, ov, dc = _overlay_plan(cfg.data.n_transfer_val, kd, cfg.data.shortcut_corr, False, gen)
        render_decoy_obs = lambda label, word: _render_decoy(label, word, sz)
        render_decoy_clean = lambda label: _render_decoy(label, None, sz)
        transfer = _encode_split_clip(
            encode_fn, y=y, overlay=ov, decoy=dc,
            render_obs=render_decoy_obs, render_clean=render_decoy_clean, num_classes=kd,
        )
        d_prompts = [cfg.data.decoy_prompt_template.format(label=name) for name in DECOY_CLASSES]
        d_text = encode_text_prompts(status, d_prompts, device).detach().cpu()
        transfer_classifier = FrozenClassifier("clip_text", cfg.model.logit_scale, text_features=d_text)

    ds = LoraCsaDataset(
        train=train, val=val, held_out_val=held_out, transfer_val=transfer,
        classifier=classifier, transfer_classifier=transfer_classifier,
        embed_dim=int(text_feats.shape[1]), num_classes=k,
        embedding_source=f"real OpenCLIP {status.model_name} / {status.pretrained_tag} image embeddings",
        backend=status.backend, clip_loaded=True,
        source_note=(
            f"real {status.backend} CLIP ({status.model_name}/{status.pretrained_tag}); "
            "controlled text-overlay shortcut images; frozen zero-shot text head; "
            "trainable adapter over frozen image embeddings"
        ),
    )
    if cfg.use_embedding_cache:
        _save_cached_dataset(cache_path, ds)
    return ds


def _save_cached_dataset(path: Path, ds: LoraCsaDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def pack(split: Optional[EmbeddingSplit]):
        if split is None:
            return None
        return {
            "observed": split.observed, "clean": split.clean, "misleading": split.misleading,
            "candidates": split.candidates, "labels": split.labels,
        }

    torch.save(
        {
            "train": pack(ds.train), "val": pack(ds.val), "held_out_val": pack(ds.held_out_val),
            "transfer_val": pack(ds.transfer_val),
            "text_features": ds.classifier.text_features,
            "transfer_text_features": ds.transfer_classifier.text_features if ds.transfer_classifier else None,
            "embed_dim": ds.embed_dim, "num_classes": ds.num_classes,
            "embedding_source": ds.embedding_source, "backend": ds.backend, "source_note": ds.source_note,
            "logit_scale": ds.classifier.logit_scale,
        },
        path,
    )


def _load_cached_dataset(path: Path, status, status_tag: str) -> LoraCsaDataset:
    blob = torch.load(path)

    def unpack(d):
        if d is None:
            return None
        return EmbeddingSplit(d["observed"], d["clean"], d["misleading"], d["candidates"], d["labels"])

    classifier = FrozenClassifier("clip_text", float(blob["logit_scale"]), text_features=blob["text_features"])
    transfer_classifier = None
    if blob.get("transfer_text_features") is not None:
        transfer_classifier = FrozenClassifier("clip_text", float(blob["logit_scale"]), text_features=blob["transfer_text_features"])
    return LoraCsaDataset(
        train=unpack(blob["train"]), val=unpack(blob["val"]), held_out_val=unpack(blob["held_out_val"]),
        transfer_val=unpack(blob["transfer_val"]), classifier=classifier, transfer_classifier=transfer_classifier,
        embed_dim=int(blob["embed_dim"]), num_classes=int(blob["num_classes"]),
        embedding_source=str(blob["embedding_source"]), backend=str(blob["backend"]), clip_loaded=True,
        source_note=str(blob["source_note"]) + " [loaded from embedding cache]",
    )


def build_synthetic_fallback(cfg: LoraPilotConfig, note: str) -> LoraCsaDataset:
    """Controlled synthetic frozen embeddings reused from the linear-head pilot.

    The trainable adapter and CSA losses are exercised identically; only the
    embedding source differs. The frozen classifier is a nearest-class-mean
    reader over the synthetic clean embeddings.
    """

    syn_cfg = SyntheticDataConfig(
        num_classes=len(SHAPE_CLASSES),
        n_train=cfg.data.n_train,
        n_val=cfg.data.n_val,
        n_held_out_val=cfg.data.n_held_out_val,
        n_transfer_val=cfg.data.n_transfer_val,
        shortcut_corr=cfg.data.shortcut_corr,
        enable_transfer_family=cfg.data.enable_transfer_family,
    )
    syn = build_synthetic_dataset(syn_cfg, cfg.seed)
    means = clean_class_means(syn.train, syn.num_classes)
    classifier = FrozenClassifier("ncm", cfg.model.logit_scale, class_means=means)
    transfer_classifier = classifier if syn.transfer_val is not None else None
    return LoraCsaDataset(
        train=syn.train, val=syn.val, held_out_val=syn.held_out_val, transfer_val=syn.transfer_val,
        classifier=classifier, transfer_classifier=transfer_classifier,
        embed_dim=syn.embed_dim, num_classes=syn.num_classes,
        embedding_source="synthetic_fallback",
        backend="synthetic", clip_loaded=False,
        source_note=note,
    )


def build_dataset(cfg: LoraPilotConfig) -> LoraCsaDataset:
    return build_clip_dataset(cfg)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def count_trainable_params(adapter: Optional[nn.Module]) -> int:
    if adapter is None:
        return 0
    return int(sum(p.numel() for p in adapter.parameters() if p.requires_grad))


def _iter_batches(n: int, batch_size: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def train_adapter(
    dataset: LoraCsaDataset,
    mode: str,
    cfg: LoraPilotConfig,
) -> Optional[nn.Module]:
    """Train an :class:`EmbeddingAdapter` under one trainable mode.

    ``mode="frozen"`` returns ``None`` (the frozen zero-shot model with no
    adapter). The clean-preservation reference for CSA is the frozen classifier's
    prediction on the training clean embeddings (no adapter), which anchors the
    adapter to the original zero-shot behaviour.
    """

    if mode == "frozen":
        return None

    gen = torch.Generator().manual_seed(cfg.seed + TRAINING_MODES.index(mode))
    adapter = EmbeddingAdapter(dataset.embed_dim, cfg.adapter.bottleneck, cfg.adapter.alpha)
    with torch.no_grad():
        adapter.down.weight.copy_(0.01 * torch.randn(adapter.down.weight.shape, generator=gen))
        adapter.down.bias.zero_()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    train = dataset.train
    clf = dataset.classifier
    reference_clean = clf.logits(train.clean).detach() if mode == "csa" else None

    for _ in range(cfg.train.epochs):
        adapter.train()
        for idx in _iter_batches(len(train), cfg.train.batch_size, gen):
            y = train.labels[idx]
            observed_logits = model_logits(adapter, clf, train.observed[idx])
            loss = F.cross_entropy(observed_logits, y)

            if mode == "cf_aug":
                cand = model_logits(adapter, clf, train.candidates[idx])
                clean_logits = model_logits(adapter, clf, train.clean[idx])
                aug = F.cross_entropy(
                    cand.reshape(-1, dataset.num_classes),
                    y.repeat_interleave(cand.shape[1]),
                )
                loss = loss + aug + F.cross_entropy(clean_logits, y)
            elif mode == "csa":
                clean_logits = model_logits(adapter, clf, train.clean[idx])
                cand = model_logits(adapter, clf, train.candidates[idx])
                misleading_logits = model_logits(adapter, clf, train.misleading[idx])
                loss, _ = csa_total_loss(
                    observed_logits, y, clean_logits, cand,
                    intervened_logits=misleading_logits,
                    reference_clean_logits=reference_clean[idx],
                    lambda_stability=cfg.csa.lambda_stability,
                    lambda_cic=cfg.csa.lambda_cic,
                    lambda_preservation=cfg.csa.lambda_preservation,
                    preservation_mode=cfg.csa.preservation_mode,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    adapter.eval()
    return adapter


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


@torch.no_grad()
def evaluate_split(adapter, classifier: FrozenClassifier, split: EmbeddingSplit) -> dict[str, float]:
    clean_logits = model_logits(adapter, classifier, split.clean)
    misleading_logits = model_logits(adapter, classifier, split.misleading)
    cand_logits = model_logits(adapter, classifier, split.candidates)  # [N, M, K]
    return {
        "clean_accuracy": _accuracy(clean_logits, split.labels),
        "shortcut_accuracy": _accuracy(misleading_logits, split.labels),
        "counterfactual_instability": float(cic_instability_penalty(cand_logits)),
    }


@torch.no_grad()
def evaluate_mode(adapter, dataset: LoraCsaDataset) -> dict[str, Any]:
    clf = dataset.classifier
    val = evaluate_split(adapter, clf, dataset.val)
    held = evaluate_split(adapter, clf, dataset.held_out_val)
    out = {
        "clean_accuracy": val["clean_accuracy"],
        "shortcut_accuracy": val["shortcut_accuracy"],
        "counterfactual_instability": val["counterfactual_instability"],
        "held_out_clean_accuracy": held["clean_accuracy"],
        "held_out_shortcut_accuracy": held["shortcut_accuracy"],
        "held_out_counterfactual_instability": held["counterfactual_instability"],
        # zero-shot/prompt accuracy on a clean held-out set (overlay-free):
        "zero_shot_clean_accuracy": held["clean_accuracy"],
    }
    if dataset.transfer_val is not None and dataset.transfer_classifier is not None:
        transfer = evaluate_split(adapter, dataset.transfer_classifier, dataset.transfer_val)
        out["transfer_clean_accuracy"] = transfer["clean_accuracy"]
        out["transfer_shortcut_accuracy"] = transfer["shortcut_accuracy"]
        out["transfer_counterfactual_instability"] = transfer["counterfactual_instability"]
    else:
        out["transfer_clean_accuracy"] = None
        out["transfer_shortcut_accuracy"] = None
        out["transfer_counterfactual_instability"] = None
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_modes(dataset: LoraCsaDataset, cfg: LoraPilotConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    results: dict[str, dict[str, Any]] = {}
    trainable_params: dict[str, int] = {}
    for mode in TRAINING_MODES:
        adapter = train_adapter(dataset, mode, cfg)
        results[mode] = evaluate_mode(adapter, dataset)
        trainable_params[mode] = count_trainable_params(adapter)
    return {"modes": results, "trainable_params": trainable_params}


def compute_go_no_go(modes: dict[str, dict[str, Any]], cfg: LoraPilotConfig) -> dict[str, Any]:
    plain = modes["plain_ft"]
    cf_aug = modes["cf_aug"]
    csa = modes["csa"]

    instab_plain = plain["counterfactual_instability"]
    instab_csa = csa["counterfactual_instability"]
    instability_drop_rel = (
        (instab_plain - instab_csa) / instab_plain if instab_plain > 1e-9 else 0.0
    )
    clean_drop_vs_plain = plain["clean_accuracy"] - csa["clean_accuracy"]
    held_gain_vs_plain = csa["held_out_shortcut_accuracy"] - plain["held_out_shortcut_accuracy"]
    held_gain_vs_cf = csa["held_out_shortcut_accuracy"] - cf_aug["held_out_shortcut_accuracy"]

    transfer_csa = csa.get("transfer_shortcut_accuracy")
    transfer_plain = plain.get("transfer_shortcut_accuracy")
    if transfer_csa is not None and transfer_plain is not None:
        transfer_delta_vs_plain = transfer_csa - transfer_plain
    else:
        transfer_delta_vs_plain = None

    instability_drop_ok = instability_drop_rel >= cfg.min_instability_drop_rel
    clean_drop_ok = clean_drop_vs_plain <= cfg.max_clean_accuracy_drop
    held_out_gain_ok = (
        held_gain_vs_plain >= cfg.min_held_out_overlay_gain
        or held_gain_vs_cf >= cfg.min_held_out_overlay_gain
    )
    # Transfer must not regress by more than the bound (no-info => not a blocker).
    transfer_ok = transfer_delta_vs_plain is None or transfer_delta_vs_plain >= -cfg.max_transfer_regression
    transfer_strong = transfer_delta_vs_plain is not None and transfer_delta_vs_plain >= cfg.strong_transfer_gain

    csa_lora_promising = bool(instability_drop_ok and clean_drop_ok and held_out_gain_ok and transfer_ok)

    return {
        "thresholds": {
            "min_instability_drop_rel": cfg.min_instability_drop_rel,
            "max_clean_accuracy_drop": cfg.max_clean_accuracy_drop,
            "min_held_out_overlay_gain": cfg.min_held_out_overlay_gain,
            "max_transfer_regression": cfg.max_transfer_regression,
            "strong_transfer_gain": cfg.strong_transfer_gain,
        },
        "instability_drop_rel_vs_plain": instability_drop_rel,
        "clean_accuracy_drop_vs_plain": clean_drop_vs_plain,
        "held_out_overlay_gain_vs_plain": held_gain_vs_plain,
        "held_out_overlay_gain_vs_cf_aug": held_gain_vs_cf,
        "transfer_delta_vs_plain": transfer_delta_vs_plain,
        "instability_drop_ok": bool(instability_drop_ok),
        "clean_drop_ok": bool(clean_drop_ok),
        "held_out_gain_ok": bool(held_out_gain_ok),
        "transfer_ok": bool(transfer_ok),
        "transfer_strong_improvement": bool(transfer_strong),
        "csa_lora_promising": csa_lora_promising,
    }


# --------------------------------------------------------------------------- #
# manual_lora_visual: actual LoRA on the OpenCLIP visual tower
# --------------------------------------------------------------------------- #
# This mode patches selected nn.Linear modules in the visual tower with the
# internal PyTorch LoRA wrapper (NO PEFT) and trains ONLY the LoRA factors. Unlike
# adapter_head_only it must forward real images through the (partly trainable)
# backbone every step, so it keeps image tensors rather than pre-encoded
# embeddings. The adapter_head_only path above is untouched.
from causal_reliability.training import manual_lora as _ml


@dataclass
class LoraImageSplit:
    """Image-space split for manual-LoRA training (uint8 tensors in [0, 255]).

    Mirrors :class:`EmbeddingSplit` but stores rendered images rather than frozen
    embeddings, because the visual tower is partly trainable. ``observed`` carries
    the painted overlay, ``clean`` removes it, ``misleading`` paints a
    competing-class overlay, and ``candidates`` stacks one image per candidate
    overlay word ([N, M, 3, H, W]) for the CIC penalty.
    """

    observed: torch.Tensor  # [N, 3, H, W] uint8
    clean: torch.Tensor
    misleading: torch.Tensor
    candidates: torch.Tensor  # [N, M, 3, H, W] uint8
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class LoraImageDataset:
    train: LoraImageSplit
    val: LoraImageSplit
    held_out_val: LoraImageSplit
    transfer_val: Optional[LoraImageSplit]
    shape_prompts: list
    decoy_prompts: Optional[list]
    num_classes: int
    num_transfer_classes: int
    image_size: int


def _stack_uint8(images) -> torch.Tensor:
    import numpy as np

    arr = (np.stack(images) * 255.0).round().clip(0, 255).astype(np.uint8)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def _build_image_split(
    *,
    y: torch.Tensor,
    overlay: torch.Tensor,
    decoy: torch.Tensor,
    render_obs,
    render_clean,
    num_classes: int,
) -> LoraImageSplit:
    n = int(y.shape[0])
    observed = _stack_uint8([render_obs(int(y[i]), int(overlay[i])) for i in range(n)])
    clean = _stack_uint8([render_clean(int(y[i])) for i in range(n)])
    misleading = _stack_uint8([render_obs(int(y[i]), int(decoy[i])) for i in range(n)])
    cand_imgs = []
    for i in range(n):
        for c in range(num_classes):
            cand_imgs.append(render_obs(int(y[i]), c))
    candidates = _stack_uint8(cand_imgs).reshape(n, num_classes, *observed.shape[1:])
    return LoraImageSplit(observed, clean, misleading, candidates, y)


def build_lora_image_dataset(cfg: LoraPilotConfig) -> LoraImageDataset:
    """Render the controlled text-overlay (and semantic-decoy) image splits.

    Returns raw image tensors (uint8) so the partly-trainable visual tower can be
    forwarded every step. Uses the same controlled stimuli as the
    ``adapter_head_only`` path.
    """

    gen = torch.Generator().manual_seed(cfg.seed)
    k = len(SHAPE_CLASSES)
    sz = cfg.data.image_size

    def render_obs_style(style):
        return lambda label, word: _render_shape_overlay(label, SHAPE_CLASSES[word], sz, style, True)

    def render_clean(label):
        return _render_shape_overlay(label, "", sz, "train", False)

    y, ov, dc = _overlay_plan(cfg.data.n_train, k, cfg.data.shortcut_corr, True, gen)
    train = _build_image_split(y=y, overlay=ov, decoy=dc, render_obs=render_obs_style("train"), render_clean=render_clean, num_classes=k)
    y, ov, dc = _overlay_plan(cfg.data.n_val, k, cfg.data.shortcut_corr, False, gen)
    val = _build_image_split(y=y, overlay=ov, decoy=dc, render_obs=render_obs_style("train"), render_clean=render_clean, num_classes=k)
    y, ov, dc = _overlay_plan(cfg.data.n_held_out_val, k, cfg.data.shortcut_corr, False, gen)
    held_out = _build_image_split(y=y, overlay=ov, decoy=dc, render_obs=render_obs_style("held_out"), render_clean=render_clean, num_classes=k)

    transfer = None
    decoy_prompts = None
    n_transfer_classes = len(DECOY_CLASSES)
    if cfg.data.enable_transfer_family:
        kd = len(DECOY_CLASSES)
        y, ov, dc = _overlay_plan(cfg.data.n_transfer_val, kd, cfg.data.shortcut_corr, False, gen)
        render_decoy_obs = lambda label, word: _render_decoy(label, word, sz)
        render_decoy_clean = lambda label: _render_decoy(label, None, sz)
        transfer = _build_image_split(y=y, overlay=ov, decoy=dc, render_obs=render_decoy_obs, render_clean=render_decoy_clean, num_classes=kd)
        decoy_prompts = [cfg.data.decoy_prompt_template.format(label=name) for name in DECOY_CLASSES]

    shape_prompts = [cfg.data.prompt_template.format(label=name) for name in SHAPE_CLASSES]
    return LoraImageDataset(
        train=train, val=val, held_out_val=held_out, transfer_val=transfer,
        shape_prompts=shape_prompts, decoy_prompts=decoy_prompts,
        num_classes=k, num_transfer_classes=n_transfer_classes, image_size=sz,
    )


# ---- model wrapper --------------------------------------------------------- #
_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalization_from_preprocess(preprocess) -> tuple:
    for t in getattr(preprocess, "transforms", []) or []:
        mean = getattr(t, "mean", None)
        std = getattr(t, "std", None)
        if mean is not None and std is not None:
            return tuple(float(x) for x in mean), tuple(float(x) for x in std)
    return _OPENAI_CLIP_MEAN, _OPENAI_CLIP_STD


class ManualLoraModel:
    """Thin wrapper turning a (LoRA-patched) visual model into a zero-shot reader.

    Holds a visual module exposing ``encode_image(x) -> [B, D]`` (the OpenCLIP
    model, or a toy stand-in in tests), the CLIP image normalization constants,
    the frozen text-prompt features for the shape and (optional) decoy families,
    and the logit scale. ``image_logits`` renders class logits from raw images in
    ``[0, 1]``. The text head stays frozen (zero-shot collapse stays measurable).
    """

    def __init__(
        self,
        visual_model,
        *,
        mean: tuple,
        std: tuple,
        logit_scale: float,
        text_features: torch.Tensor,
        transfer_text_features: Optional[torch.Tensor],
        encode_batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        self.model = visual_model
        self.device = device
        self.logit_scale = float(logit_scale)
        self.text_features = text_features
        self.transfer_text_features = transfer_text_features
        self.encode_batch_size = int(max(1, encode_batch_size))
        self._mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def reset_lora(self, seed: Optional[int] = None) -> None:
        _ml.reset_all_lora_parameters(self.model, seed=seed)

    def _to_float01(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        return images

    def _encode(self, images: torch.Tensor, grad: bool) -> torch.Tensor:
        images = self._to_float01(images)
        feats = []
        bs = self.encode_batch_size
        for i in range(0, len(images), bs):
            chunk = (images[i : i + bs].to(self.device) - self._mean) / self._std
            if grad:
                out = self.model.encode_image(chunk)
            else:
                with torch.no_grad():
                    out = self.model.encode_image(chunk)
            feats.append(F.normalize(out, dim=-1))
        return torch.cat(feats, dim=0)

    def image_logits(self, images: torch.Tensor, classifier: str = "shape", grad: bool = False) -> torch.Tensor:
        feats = self._encode(images, grad=grad)
        tf = self.transfer_text_features if classifier == "transfer" else self.text_features
        return self.logit_scale * (feats @ tf.T)

    def candidate_logits(self, images: torch.Tensor, classifier: str = "shape", grad: bool = False) -> torch.Tensor:
        n, m = images.shape[0], images.shape[1]
        flat = images.reshape(n * m, *images.shape[2:])
        logits = self.image_logits(flat, classifier=classifier, grad=grad)
        return logits.reshape(n, m, -1)


def build_real_lora_model(cfg: LoraPilotConfig) -> dict[str, Any]:
    """Load real OpenCLIP, patch the visual tower with LoRA, and build the reader.

    Returns ``{"ok": False, "reason": ...}`` if OpenCLIP cannot be loaded
    (skip cleanly). On success returns ``{"ok": True, "model": ManualLoraModel,
    "patch_info": ..., "backend": ..., ...}``. Raises only if patching genuinely
    finds zero target modules (a real misconfiguration).
    """

    from causal_reliability.real_models.clip_zero_shot import (
        check_clip_available,
        encode_text_prompts,
    )

    status = check_clip_available(
        device=cfg.model.device,
        allow_download=cfg.model.allow_download,
        preferred_backend=cfg.model.backend,
        model_name=cfg.model.model_name,
        pretrained_tag=cfg.model.pretrained_tag,
        transformers_model_name=cfg.model.transformers_model_name,
    )
    if not (status.available and status.backend == "open_clip" and status.pretrained):
        return {
            "ok": False,
            "reason": (
                "manual_lora_visual requires a loadable OpenCLIP (open_clip) backbone "
                "to patch its visual tower; "
                f"{status.error_message or 'OpenCLIP unavailable'}. Skipping cleanly."
            ),
            "backend": status.backend,
        }

    mlc = cfg.manual_lora
    patch_info = _ml.apply_lora_to_openclip_visual(
        status.model,
        target_modules=mlc.target_modules,
        rank=mlc.rank,
        alpha=mlc.alpha,
        dropout=mlc.dropout,
        max_layers=mlc.target_last_blocks,
    )

    shape_prompts = [cfg.data.prompt_template.format(label=name) for name in SHAPE_CLASSES]
    text_features = encode_text_prompts(status, shape_prompts, cfg.model.device).detach().cpu()
    transfer_text_features = None
    if cfg.data.enable_transfer_family:
        decoy_prompts = [cfg.data.decoy_prompt_template.format(label=name) for name in DECOY_CLASSES]
        transfer_text_features = encode_text_prompts(status, decoy_prompts, cfg.model.device).detach().cpu()

    mean, std = _normalization_from_preprocess(status.preprocess)
    model = ManualLoraModel(
        status.model,
        mean=mean,
        std=std,
        logit_scale=cfg.model.logit_scale,
        text_features=text_features,
        transfer_text_features=transfer_text_features,
        encode_batch_size=mlc.encode_batch_size,
        device=cfg.model.device,
    )
    return {
        "ok": True,
        "model": model,
        "patch_info": patch_info,
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_tag": status.pretrained_tag,
    }


# ---- training / evaluation ------------------------------------------------- #
def train_manual_lora(model: ManualLoraModel, dataset: LoraImageDataset, mode: str, cfg: LoraPilotConfig) -> None:
    """Train ONLY the LoRA factors under one mode (in place on ``model``).

    ``mode="frozen"`` resets the LoRA factors to identity and does no training, so
    the model reproduces the frozen zero-shot backbone exactly.
    """

    seed = cfg.seed + TRAINING_MODES.index(mode)
    model.reset_lora(seed=seed)
    if mode == "frozen":
        return

    mlc = cfg.manual_lora
    params = model.trainable_parameters()
    optimizer = torch.optim.Adam(params, lr=mlc.lr, weight_decay=0.0)

    train = dataset.train
    k = dataset.num_classes
    # Reference for clean-preservation: the frozen zero-shot head's clean logits,
    # captured at LoRA identity (i.e. the frozen model) before any update.
    reference_clean = (
        model.image_logits(train.clean, classifier="shape", grad=False).detach()
        if mode == "csa"
        else None
    )

    gen = torch.Generator().manual_seed(seed)
    n = len(train)
    for _ in range(mlc.epochs):
        for idx in _iter_batches(n, mlc.batch_size, gen):
            y = train.labels[idx]
            observed_logits = model.image_logits(train.observed[idx], grad=True)
            loss = F.cross_entropy(observed_logits, y)

            if mode == "cf_aug":
                clean_logits = model.image_logits(train.clean[idx], grad=True)
                cand = model.candidate_logits(train.candidates[idx], grad=True)
                aug = F.cross_entropy(cand.reshape(-1, k), y.repeat_interleave(cand.shape[1]))
                loss = loss + aug + F.cross_entropy(clean_logits, y)
            elif mode == "csa":
                clean_logits = model.image_logits(train.clean[idx], grad=True)
                cand = model.candidate_logits(train.candidates[idx], grad=True)
                misleading_logits = model.image_logits(train.misleading[idx], grad=True)
                loss, _ = csa_total_loss(
                    observed_logits, y, clean_logits, cand,
                    intervened_logits=misleading_logits,
                    reference_clean_logits=reference_clean[idx],
                    lambda_stability=cfg.csa.lambda_stability,
                    lambda_cic=cfg.csa.lambda_cic,
                    lambda_preservation=cfg.csa.lambda_preservation,
                    preservation_mode=cfg.csa.preservation_mode,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_manual_lora_split(model: ManualLoraModel, split: LoraImageSplit, classifier: str) -> dict[str, float]:
    clean_logits = model.image_logits(split.clean, classifier=classifier, grad=False)
    misleading_logits = model.image_logits(split.misleading, classifier=classifier, grad=False)
    cand_logits = model.candidate_logits(split.candidates, classifier=classifier, grad=False)
    return {
        "clean_accuracy": _accuracy(clean_logits, split.labels),
        "shortcut_accuracy": _accuracy(misleading_logits, split.labels),
        "counterfactual_instability": float(cic_instability_penalty(cand_logits)),
    }


@torch.no_grad()
def evaluate_manual_lora_mode(model: ManualLoraModel, dataset: LoraImageDataset) -> dict[str, Any]:
    val = evaluate_manual_lora_split(model, dataset.val, "shape")
    held = evaluate_manual_lora_split(model, dataset.held_out_val, "shape")
    out = {
        "clean_accuracy": val["clean_accuracy"],
        "shortcut_accuracy": val["shortcut_accuracy"],
        "counterfactual_instability": val["counterfactual_instability"],
        "held_out_clean_accuracy": held["clean_accuracy"],
        "held_out_shortcut_accuracy": held["shortcut_accuracy"],
        "held_out_counterfactual_instability": held["counterfactual_instability"],
        "zero_shot_clean_accuracy": held["clean_accuracy"],
    }
    if dataset.transfer_val is not None and model.transfer_text_features is not None:
        transfer = evaluate_manual_lora_split(model, dataset.transfer_val, "transfer")
        out["transfer_clean_accuracy"] = transfer["clean_accuracy"]
        out["transfer_shortcut_accuracy"] = transfer["shortcut_accuracy"]
        out["transfer_counterfactual_instability"] = transfer["counterfactual_instability"]
    else:
        out["transfer_clean_accuracy"] = None
        out["transfer_shortcut_accuracy"] = None
        out["transfer_counterfactual_instability"] = None
    return out


def run_manual_lora_modes(model: ManualLoraModel, dataset: LoraImageDataset, cfg: LoraPilotConfig, lora_param_count: int) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    results: dict[str, dict[str, Any]] = {}
    trainable_params: dict[str, int] = {}
    for mode in TRAINING_MODES:
        train_manual_lora(model, dataset, mode, cfg)
        results[mode] = evaluate_manual_lora_mode(model, dataset)
        trainable_params[mode] = 0 if mode == "frozen" else int(lora_param_count)
    return {"modes": results, "trainable_params": trainable_params}


def compute_manual_lora_go_no_go(modes: dict[str, dict[str, Any]], cfg: LoraPilotConfig) -> dict[str, Any]:
    """Pre-registered, stronger go/no-go for the manual_lora_visual mode.

    ``manual_lora_promising = true`` only if ALL hold:
      1. CSA reduces CIC instability >= 20% relative to plain manual-LoRA.
      2. clean accuracy drop relative to FROZEN zero-shot <= 0.03.
      3. held-out text-overlay accuracy improves >= +0.05 over plain manual-LoRA.
      4. CSA is not worse than cf_aug on held-out text overlays by more than 0.02.
      5. semantic-decoy transfer is not worse than plain by more than 0.03.
    ``manual_lora_strong = true`` only if CSA beats cf_aug by >= +0.05 on either
    held-out text overlays or semantic-decoy transfer while preserving clean
    accuracy within 0.03 of frozen.
    """

    frozen, plain, cf_aug, csa = modes["frozen"], modes["plain_ft"], modes["cf_aug"], modes["csa"]

    instab_plain = plain["counterfactual_instability"]
    instab_csa = csa["counterfactual_instability"]
    instability_drop_rel = (instab_plain - instab_csa) / instab_plain if instab_plain > 1e-9 else 0.0

    clean_drop_vs_frozen = frozen["clean_accuracy"] - csa["clean_accuracy"]
    held_gain_vs_plain = csa["held_out_shortcut_accuracy"] - plain["held_out_shortcut_accuracy"]
    held_delta_vs_cf = csa["held_out_shortcut_accuracy"] - cf_aug["held_out_shortcut_accuracy"]

    transfer_csa = csa.get("transfer_shortcut_accuracy")
    transfer_plain = plain.get("transfer_shortcut_accuracy")
    transfer_cf = cf_aug.get("transfer_shortcut_accuracy")
    transfer_delta_vs_plain = (
        transfer_csa - transfer_plain if (transfer_csa is not None and transfer_plain is not None) else None
    )
    transfer_delta_vs_cf = (
        transfer_csa - transfer_cf if (transfer_csa is not None and transfer_cf is not None) else None
    )

    instability_drop_ok = instability_drop_rel >= cfg.ml_min_instability_drop_rel
    clean_drop_ok = clean_drop_vs_frozen <= cfg.ml_max_clean_drop_vs_frozen
    held_out_gain_ok = held_gain_vs_plain >= cfg.ml_min_held_out_gain_vs_plain
    held_vs_cf_ok = held_delta_vs_cf >= -cfg.ml_max_held_out_regression_vs_cf
    transfer_ok = transfer_delta_vs_plain is None or transfer_delta_vs_plain >= -cfg.ml_max_transfer_regression_vs_plain

    manual_lora_promising = bool(
        instability_drop_ok and clean_drop_ok and held_out_gain_ok and held_vs_cf_ok and transfer_ok
    )

    beats_cf_held = held_delta_vs_cf >= cfg.ml_strong_cf_gain
    beats_cf_transfer = transfer_delta_vs_cf is not None and transfer_delta_vs_cf >= cfg.ml_strong_cf_gain
    manual_lora_strong = bool((beats_cf_held or beats_cf_transfer) and clean_drop_ok)

    return {
        "thresholds": {
            "min_instability_drop_rel": cfg.ml_min_instability_drop_rel,
            "max_clean_drop_vs_frozen": cfg.ml_max_clean_drop_vs_frozen,
            "min_held_out_gain_vs_plain": cfg.ml_min_held_out_gain_vs_plain,
            "max_held_out_regression_vs_cf": cfg.ml_max_held_out_regression_vs_cf,
            "max_transfer_regression_vs_plain": cfg.ml_max_transfer_regression_vs_plain,
            "strong_cf_gain": cfg.ml_strong_cf_gain,
        },
        "instability_drop_rel_vs_plain": instability_drop_rel,
        "clean_accuracy_drop_vs_frozen": clean_drop_vs_frozen,
        "held_out_overlay_gain_vs_plain": held_gain_vs_plain,
        "held_out_overlay_delta_vs_cf_aug": held_delta_vs_cf,
        "transfer_delta_vs_plain": transfer_delta_vs_plain,
        "transfer_delta_vs_cf_aug": transfer_delta_vs_cf,
        "instability_drop_ok": bool(instability_drop_ok),
        "clean_drop_ok": bool(clean_drop_ok),
        "held_out_gain_ok": bool(held_out_gain_ok),
        "held_out_vs_cf_ok": bool(held_vs_cf_ok),
        "transfer_ok": bool(transfer_ok),
        "manual_lora_promising": manual_lora_promising,
        "manual_lora_strong": manual_lora_strong,
    }


__all__ = [
    "TRAINING_MODES",
    "SHAPE_CLASSES",
    "DECOY_CLASSES",
    "LoraDataConfig",
    "LoraTrainConfig",
    "AdapterConfig",
    "LoraCsaConfig",
    "ManualLoraConfig",
    "ModelConfig",
    "LoraPilotConfig",
    "EmbeddingAdapter",
    "FrozenClassifier",
    "LoraCsaDataset",
    "lora_dependencies_available",
    "resolve_trainable_module",
    "model_logits",
    "build_dataset",
    "build_clip_dataset",
    "build_synthetic_fallback",
    "train_adapter",
    "count_trainable_params",
    "evaluate_split",
    "evaluate_mode",
    "run_modes",
    "compute_go_no_go",
    "LoraImageSplit",
    "LoraImageDataset",
    "build_lora_image_dataset",
    "ManualLoraModel",
    "build_real_lora_model",
    "train_manual_lora",
    "evaluate_manual_lora_split",
    "evaluate_manual_lora_mode",
    "run_manual_lora_modes",
    "compute_manual_lora_go_no_go",
]
