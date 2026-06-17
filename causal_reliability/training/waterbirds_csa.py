from __future__ import annotations

"""Pre-registered Waterbirds CSA manual-LoRA pilot.

Bounded scientific question
---------------------------
Does Counterfactual Stability Alignment (CSA) applied as a small manual-LoRA
adaptation of a real OpenCLIP visual tower improve **worst-group robustness** on
the real WILDS Waterbirds dataset *without using group labels for CSA training*?

This is **one bounded experiment**, not a benchmark search. A positive, null, or
negative result is all acceptable and is reported honestly.

Scope / explicit non-claims
---------------------------
This pilot is **not** universal robustness, **not** open-world shortcut
discovery, **not** an RLHF/DPO replacement, **not** deployment validation, and
**not** a replacement for the finalized STS report.

Honesty rules baked into the code
----------------------------------
* Group labels (``y`` × ``place``) are used **only** for evaluation (per-group /
  worst-group accuracy) and for an *optional, clearly-marked* Group DRO baseline.
  They are never used by the CSA training objective.
* The CSA interventions for Waterbirds are **finite diagnostic interventions**
  (label-free background/region perturbations), **not** verified ground-truth
  Waterbirds causal masks. The summary states this explicitly.
* Real ``manual_lora_visual`` requires CUDA/MPS and a loadable OpenCLIP backbone.
  If neither GPU/MPS nor OpenCLIP is available the pilot falls back to a labelled
  ``cached_embedding_adapter`` (lightweight head over FROZEN CLIP image
  embeddings — **not** LoRA). The fallback is diagnostic only and can **never**
  set ``waterbirds_csa_promising`` or ``waterbirds_csa_strong`` to true.

Reused building blocks
----------------------
* :func:`causal_reliability.training.manual_lora.apply_lora_to_openclip_visual`
  and :class:`~causal_reliability.training.manual_lora.LoRALinear` for the real
  LoRA patch.
* :class:`causal_reliability.training.csa_lora.ManualLoraModel` /
  :class:`~causal_reliability.training.csa_lora.EmbeddingAdapter` /
  :class:`~causal_reliability.training.csa_lora.FrozenClassifier` for the
  zero-shot reader and the fallback adapter.
* :func:`causal_reliability.training.csa_loss.csa_total_loss` and
  :func:`~causal_reliability.training.csa_loss.cic_instability_penalty` for the
  CSA objective and the CIC instability metric.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_reliability.training import manual_lora as _ml
from causal_reliability.training.csa_loss import cic_instability_penalty, csa_total_loss
from causal_reliability.training.csa_lora import (
    EmbeddingAdapter,
    FrozenClassifier,
    ManualLoraModel,
    _normalization_from_preprocess,
    model_logits,
)


# Binary object label (the causal content): WILDS Waterbirds uses y=0 landbird,
# y=1 waterbird. ``place`` is the spurious background: 0 land, 1 water.
WATERBIRD_CLASSES = ["landbird", "waterbird"]

# The four evaluation groups in the requested order.
GROUP_NAMES = [
    "landbird_on_land",
    "landbird_on_water",
    "waterbird_on_land",
    "waterbird_on_water",
]

# Training modes. ``group_dro`` is optional and group-label-supervised.
BASE_MODES = ("frozen", "plain_ft", "csa")
OPTIONAL_MODES = ("cf_aug", "group_dro")

# Finite, conservatively-defined diagnostic interventions (label-free).
INTERVENTION_TYPES = ("region_mask", "region_blur", "lowfreq_bg", "weak_crop")

SKIP_NO_DATA_MESSAGE = "Waterbirds dataset unavailable; skipping cleanly"


def group_index(y: int, place: int) -> int:
    """Map (object label, background place) to a group index in ``GROUP_NAMES``."""

    return int(y) * 2 + int(place)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class WBModelConfig:
    backend: str = "open_clip"
    model_name: str = "ViT-B-32"
    pretrained_tag: str = "laion2b_s34b_b79k"
    transformers_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"  # auto -> cuda > mps > cpu
    allow_download: bool = False
    logit_scale: float = 100.0


@dataclass
class WBDataConfig:
    wilds_root: str = "data/wilds"
    wilds_dataset: str = "waterbirds"
    image_size: int = 224
    max_train_examples: int = 256
    max_eval_examples: int = 256
    eval_splits: tuple = ("val", "test")
    primary_eval_split: str = "test"
    landbird_prompt: str = "a photo of a landbird"
    waterbird_prompt: str = "a photo of a waterbird"
    download: bool = False


@dataclass
class WBInterventionConfig:
    """Finite, label-free diagnostic interventions for the CSA / CIC signal.

    These are *not* verified Waterbirds causal masks; they are conservative
    background/region perturbations used to build a finite candidate bank.
    """

    n_candidates: int = 4
    types: tuple = INTERVENTION_TYPES
    mask_frac: float = 0.35       # side fraction of the masked/blurred rectangle
    blur_kernel: int = 9          # average-pool kernel for region blur
    lowfreq_scale: float = 0.15   # amplitude of the low-frequency background field
    lowfreq_grid: int = 8         # coarse grid size upsampled to image size
    crop_min: float = 0.85        # weak object-preserving crop lower bound
    seed: int = 12345


@dataclass
class WBManualLoraConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_last_blocks: int = 2
    target_modules: tuple = ("c_fc", "c_proj", "out_proj")
    lr: float = 2e-4
    epochs: int = 2
    batch_size: int = 16
    encode_batch_size: int = 32
    train_text: bool = False  # text tower stays frozen

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "WBManualLoraConfig":
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
            epochs=int(raw.get("epochs", 2)),
            batch_size=int(raw.get("batch_size", 16)),
            encode_batch_size=int(raw.get("encode_batch_size", 32)),
            train_text=bool(raw.get("train_text", False)),
        )


@dataclass
class WBCsaConfig:
    lambda_stability: float = 1.0
    lambda_cic: float = 1.0
    lambda_preserve: float = 0.5
    preservation_mode: str = "kl"


@dataclass
class WBFallbackConfig:
    """Lightweight adapter/head over FROZEN embeddings (the CPU-safe fallback)."""

    bottleneck: int = 32
    alpha: float = 1.0
    lr: float = 0.01
    epochs: int = 3
    batch_size: int = 32


@dataclass
class WBGroupDROConfig:
    enabled: bool = False  # off by default; uses group labels (clearly supervised)
    lr: float = 2e-4
    eta: float = 0.01      # adversarial group-weight step size


@dataclass
class WBPilotConfig:
    seed: int = 0
    model: WBModelConfig = field(default_factory=WBModelConfig)
    data: WBDataConfig = field(default_factory=WBDataConfig)
    interventions: WBInterventionConfig = field(default_factory=WBInterventionConfig)
    manual_lora: WBManualLoraConfig = field(default_factory=WBManualLoraConfig)
    csa: WBCsaConfig = field(default_factory=WBCsaConfig)
    fallback: WBFallbackConfig = field(default_factory=WBFallbackConfig)
    group_dro: WBGroupDROConfig = field(default_factory=WBGroupDROConfig)
    cache_dir: str = "results/csa_lora_pilot/waterbirds/cache"
    use_embedding_cache: bool = True
    force_cpu_lora: bool = False  # allow (slow) manual-LoRA on CPU only if explicit
    enable_cf_aug: bool = False   # optional counterfactual-augmentation baseline

    # Pre-registered go / no-go thresholds.
    max_avg_acc_drop: float = 0.03         # CSA avg acc must not drop > 0.03 vs plain
    min_worst_group_gain: float = 0.05     # CSA worst-group must beat plain by >= +0.05
    min_instability_drop_rel: float = 0.20  # CSA CIC instability >= 20% below plain
    strong_worst_group_gain: float = 0.08  # multi-seed strong threshold

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WBPilotConfig":
        raw = dict(raw or {})
        gng = dict(raw.get("go_no_go", {}) or {})
        data_raw = dict(raw.get("data", {}) or {})
        if "eval_splits" in data_raw and data_raw["eval_splits"] is not None:
            data_raw["eval_splits"] = tuple(data_raw["eval_splits"])
        iv_raw = dict(raw.get("interventions", {}) or {})
        if "types" in iv_raw and iv_raw["types"] is not None:
            iv_raw["types"] = tuple(iv_raw["types"])
        return cls(
            seed=int(raw.get("seed", 0)),
            model=WBModelConfig(**(raw.get("model", {}) or {})),
            data=WBDataConfig(**data_raw),
            interventions=WBInterventionConfig(**iv_raw),
            manual_lora=WBManualLoraConfig.from_raw(raw.get("manual_lora", {})),
            csa=WBCsaConfig(**(raw.get("csa", {}) or {})),
            fallback=WBFallbackConfig(**(raw.get("fallback", {}) or {})),
            group_dro=WBGroupDROConfig(**(raw.get("group_dro", {}) or {})),
            cache_dir=str(raw.get("cache_dir", "results/csa_lora_pilot/waterbirds/cache")),
            use_embedding_cache=bool(raw.get("use_embedding_cache", True)),
            force_cpu_lora=bool(raw.get("force_cpu_lora", False)),
            enable_cf_aug=bool(raw.get("enable_cf_aug", False)),
            max_avg_acc_drop=float(gng.get("max_avg_acc_drop", 0.03)),
            min_worst_group_gain=float(gng.get("min_worst_group_gain", 0.05)),
            min_instability_drop_rel=float(gng.get("min_instability_drop_rel", 0.20)),
            strong_worst_group_gain=float(gng.get("strong_worst_group_gain", 0.08)),
        )

    def training_modes(self) -> list[str]:
        modes = list(BASE_MODES)
        if self.enable_cf_aug:
            modes.insert(2, "cf_aug")  # frozen, plain_ft, cf_aug, csa
        if self.group_dro.enabled:
            modes.append("group_dro")
        return modes


# --------------------------------------------------------------------------- #
# Device resolution
# --------------------------------------------------------------------------- #
def detect_device(requested: str = "auto") -> str:
    """Resolve a device string. ``auto`` prefers CUDA, then MPS, then CPU."""

    req = (requested or "auto").strip().lower()
    if req not in ("auto", ""):
        return req
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def is_accelerator(device: str) -> bool:
    return device.split(":")[0] in ("cuda", "mps")


def decide_mode(cfg: WBPilotConfig, device: str) -> str:
    """``manual_lora_visual`` only on an accelerator (or forced); else fallback."""

    if is_accelerator(device) or cfg.force_cpu_lora:
        return "manual_lora_visual"
    return "cached_embedding_adapter"


# --------------------------------------------------------------------------- #
# Dataset availability + loading (group labels used ONLY for eval)
# --------------------------------------------------------------------------- #
def waterbirds_dataset_dir(cfg: WBPilotConfig, download: bool = False) -> Optional[Path]:
    """Locate a local WILDS Waterbirds release dir (``*_v*`` with metadata.csv)."""

    root = Path(cfg.data.wilds_root)
    name = cfg.data.wilds_dataset
    for cand in sorted(root.glob(f"{name}_v*")):
        if (cand / "metadata.csv").exists():
            return cand
    if (root / "metadata.csv").exists():
        return root
    if download:  # pragma: no cover - exercised only with wilds installed + --download
        try:
            from wilds import get_dataset

            ds = get_dataset(dataset=name, download=True, root_dir=str(root))
            data_dir = Path(getattr(ds, "_data_dir", getattr(ds, "data_dir", root)))
            if (data_dir / "metadata.csv").exists():
                return data_dir
        except Exception:
            return None
    return None


def check_waterbirds_available(cfg: WBPilotConfig, download: bool = False) -> dict[str, Any]:
    """Structured availability report. Never raises on missing data."""

    report: dict[str, Any] = {
        "waterbirds_available": False,
        "root": "",
        "metadata_csv": "",
        "n_rows": 0,
        "reason": "",
    }
    data_dir = waterbirds_dataset_dir(cfg, download=download)
    if data_dir is None:
        report["reason"] = (
            f"{SKIP_NO_DATA_MESSAGE} (no WILDS {cfg.data.wilds_dataset} release under "
            f"{cfg.data.wilds_root})"
        )
        return report

    metadata_csv = data_dir / "metadata.csv"
    report["root"] = str(data_dir)
    report["metadata_csv"] = str(metadata_csv)
    try:
        import pandas as pd

        df = pd.read_csv(metadata_csv)
    except Exception as exc:  # pragma: no cover - defensive
        report["reason"] = f"{SKIP_NO_DATA_MESSAGE} (unreadable metadata: {exc})"
        return report

    for col in ("img_filename", "y", "split", "place"):
        if col not in df.columns:
            report["reason"] = f"{SKIP_NO_DATA_MESSAGE} (missing '{col}' column)"
            return report

    sample = df.head(32)
    present = any((data_dir / str(v)).exists() for v in sample["img_filename"].tolist())
    if not present:
        report["reason"] = f"{SKIP_NO_DATA_MESSAGE} (no image files found under {data_dir})"
        return report

    report["waterbirds_available"] = True
    report["n_rows"] = int(len(df))
    report["reason"] = "available"
    return report


@dataclass
class WBImageSplit:
    images: torch.Tensor   # [N, 3, H, W] uint8
    labels: torch.Tensor   # [N] object label y (0 landbird / 1 waterbird)
    places: torch.Tensor   # [N] background place (0 land / 1 water)
    groups: torch.Tensor   # [N] group index in GROUP_NAMES
    candidates: Optional[torch.Tensor] = None  # [N, M, 3, H, W] uint8

    def __len__(self) -> int:
        return int(self.labels.shape[0])


_SPLIT_CODE = {"train": 0, "val": 1, "validation": 1, "test": 2}


def _load_split_images(
    data_dir: Path, df, split: str, size: int, max_n: int, seed: int
) -> Optional[WBImageSplit]:
    import numpy as np
    from PIL import Image

    code = _SPLIT_CODE.get(split.lower())
    if code is None or "split" not in df.columns:
        return None
    sub = df[df["split"] == code].reset_index(drop=True)
    if len(sub) == 0:
        return None
    if len(sub) > max_n:
        sub = sub.sample(n=max_n, random_state=seed).reset_index(drop=True)

    imgs, ys, places = [], [], []
    for _, row in sub.iterrows():
        path = data_dir / str(row["img_filename"])
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        except Exception:  # pragma: no cover - defensive
            continue
        arr = (np.asarray(img)).astype("uint8")
        imgs.append(arr)
        ys.append(int(row["y"]))
        places.append(int(row["place"]))
    if not imgs:
        return None
    images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # uint8
    labels = torch.tensor(ys, dtype=torch.long)
    place_t = torch.tensor(places, dtype=torch.long)
    groups = labels * 2 + place_t
    return WBImageSplit(images=images, labels=labels, places=place_t, groups=groups)


@dataclass
class WaterbirdsDataset:
    train: WBImageSplit
    eval_splits: dict[str, WBImageSplit]
    primary_eval_split: str
    num_classes: int
    image_size: int
    intervention_summary: dict[str, Any]


def load_waterbirds_dataset(cfg: WBPilotConfig, avail: dict[str, Any]) -> WaterbirdsDataset:
    """Load the train split and the requested eval splits, attaching candidates."""

    import pandas as pd

    data_dir = Path(avail["root"])
    df = pd.read_csv(avail["metadata_csv"])
    size = cfg.data.image_size

    train = _load_split_images(data_dir, df, "train", size, cfg.data.max_train_examples, cfg.seed)
    if train is None:
        raise RuntimeError("Waterbirds train split could not be loaded")
    train.candidates = build_candidate_interventions(train.images, cfg.interventions)

    eval_splits: dict[str, WBImageSplit] = {}
    for split in cfg.data.eval_splits:
        s = _load_split_images(data_dir, df, split, size, cfg.data.max_eval_examples, cfg.seed + 1)
        if s is not None:
            s.candidates = build_candidate_interventions(s.images, cfg.interventions)
            eval_splits[split] = s

    primary = cfg.data.primary_eval_split
    if primary not in eval_splits:
        primary = next(iter(eval_splits)) if eval_splits else primary

    return WaterbirdsDataset(
        train=train,
        eval_splits=eval_splits,
        primary_eval_split=primary,
        num_classes=len(WATERBIRD_CLASSES),
        image_size=size,
        intervention_summary=describe_interventions(cfg.interventions),
    )


# --------------------------------------------------------------------------- #
# Finite diagnostic interventions (label-free; NOT verified causal masks)
# --------------------------------------------------------------------------- #
def describe_interventions(icfg: WBInterventionConfig) -> dict[str, Any]:
    chosen = [icfg.types[i % len(icfg.types)] for i in range(icfg.n_candidates)]
    return {
        "n_candidates": int(icfg.n_candidates),
        "candidate_types": chosen,
        "available_types": list(icfg.types),
        "mask_frac": icfg.mask_frac,
        "blur_kernel": icfg.blur_kernel,
        "lowfreq_scale": icfg.lowfreq_scale,
        "crop_min": icfg.crop_min,
        "honesty_note": (
            "Finite diagnostic interventions only: label-free background/region "
            "perturbations (random rectangular masking/blurring, low-frequency "
            "background perturbation, bounded object-preserving weak crops). These "
            "are NOT verified ground-truth Waterbirds causal masks and do not use "
            "group labels."
        ),
    }


def _apply_one_intervention(img: torch.Tensor, kind: str, icfg: WBInterventionConfig, gen: torch.Generator) -> torch.Tensor:
    """Apply a single label-free intervention to a float image ``[3, H, W]`` in [0, 1]."""

    c, h, w = img.shape
    if kind == "region_mask":
        rh, rw = int(h * icfg.mask_frac), int(w * icfg.mask_frac)
        rh, rw = max(1, rh), max(1, rw)
        y0 = int(torch.randint(0, max(1, h - rh + 1), (1,), generator=gen).item())
        x0 = int(torch.randint(0, max(1, w - rw + 1), (1,), generator=gen).item())
        out = img.clone()
        out[:, y0 : y0 + rh, x0 : x0 + rw] = 0.5
        return out
    if kind == "region_blur":
        k = max(3, int(icfg.blur_kernel) | 1)  # odd
        pad = k // 2
        blurred = F.avg_pool2d(img.unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0)
        rh, rw = max(1, int(h * icfg.mask_frac)), max(1, int(w * icfg.mask_frac))
        y0 = int(torch.randint(0, max(1, h - rh + 1), (1,), generator=gen).item())
        x0 = int(torch.randint(0, max(1, w - rw + 1), (1,), generator=gen).item())
        out = img.clone()
        out[:, y0 : y0 + rh, x0 : x0 + rw] = blurred[:, y0 : y0 + rh, x0 : x0 + rw]
        return out
    if kind == "lowfreq_bg":
        g = max(2, int(icfg.lowfreq_grid))
        coarse = torch.randn(1, c, g, g, generator=gen)
        field = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        return (img + icfg.lowfreq_scale * field).clamp(0.0, 1.0)
    if kind == "weak_crop":
        frac = icfg.crop_min + (1.0 - icfg.crop_min) * float(torch.rand(1, generator=gen).item())
        ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
        y0 = int(torch.randint(0, max(1, h - ch + 1), (1,), generator=gen).item())
        x0 = int(torch.randint(0, max(1, w - cw + 1), (1,), generator=gen).item())
        crop = img[:, y0 : y0 + ch, x0 : x0 + cw].unsqueeze(0)
        return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    raise ValueError(f"unknown intervention type: {kind!r}")


def build_candidate_interventions(images_uint8: torch.Tensor, icfg: WBInterventionConfig) -> torch.Tensor:
    """Build the finite candidate bank ``[N, M, 3, H, W]`` (uint8) for each image.

    Interventions are label-free and deterministic given ``icfg.seed`` so the same
    bank is shared across all training modes and across evaluation.
    """

    n = images_uint8.shape[0]
    floats = images_uint8.float() / 255.0
    chosen = [icfg.types[i % len(icfg.types)] for i in range(icfg.n_candidates)]
    out = torch.empty(n, len(chosen), *images_uint8.shape[1:], dtype=torch.uint8)
    for i in range(n):
        for m, kind in enumerate(chosen):
            gen = torch.Generator().manual_seed(int(icfg.seed) + i * 131 + m * 17)
            perturbed = _apply_one_intervention(floats[i], kind, icfg, gen)
            out[i, m] = (perturbed.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return out


# --------------------------------------------------------------------------- #
# Real OpenCLIP manual-LoRA model
# --------------------------------------------------------------------------- #
def build_real_lora_model(cfg: WBPilotConfig, device: str) -> dict[str, Any]:
    """Load real OpenCLIP, patch the visual tower with LoRA, build a binary reader.

    Returns ``{"ok": False, "reason": ...}`` when OpenCLIP cannot be loaded so the
    caller can skip cleanly.
    """

    from causal_reliability.real_models.clip_zero_shot import (
        check_clip_available,
        encode_text_prompts,
    )

    status = check_clip_available(
        device=device,
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
                f"to patch its visual tower; {status.error_message or 'OpenCLIP unavailable'}."
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

    prompts = [cfg.data.landbird_prompt, cfg.data.waterbird_prompt]
    text_features = encode_text_prompts(status, prompts, device).detach().to(device)
    mean, std = _normalization_from_preprocess(status.preprocess)
    model = ManualLoraModel(
        status.model,
        mean=mean,
        std=std,
        logit_scale=cfg.model.logit_scale,
        text_features=text_features,
        transfer_text_features=None,
        encode_batch_size=mlc.encode_batch_size,
        device=device,
    )
    # ManualLoraModel keeps the normalization buffers on CPU by default; move them
    # to the compute device so the partly-trainable forward stays on-device.
    model._mean = model._mean.to(device)
    model._std = model._std.to(device)
    return {
        "ok": True,
        "model": model,
        "patch_info": patch_info,
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_tag": status.pretrained_tag,
        "status": status,
    }


def _iter_batches(n: int, batch_size: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def train_manual_lora_mode(model: ManualLoraModel, dataset: WaterbirdsDataset, mode: str, cfg: WBPilotConfig) -> None:
    """Train ONLY the LoRA factors under one mode (in place). ``frozen`` resets to identity.

    No mode uses group labels except ``group_dro`` (explicitly supervised baseline).
    """

    seed = cfg.seed + (cfg.training_modes().index(mode) if mode in cfg.training_modes() else 0)
    model.reset_lora(seed=seed)
    if mode == "frozen":
        return

    mlc = cfg.manual_lora
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=mlc.lr, weight_decay=0.0)
    train = dataset.train
    k = dataset.num_classes
    reference_clean = (
        model.image_logits(train.images, classifier="shape", grad=False).detach()
        if mode == "csa"
        else None
    )

    gen = torch.Generator().manual_seed(seed)
    n = len(train)
    if mode == "group_dro":
        group_weights = torch.ones(len(GROUP_NAMES)) / len(GROUP_NAMES)

    for _ in range(mlc.epochs):
        for idx in _iter_batches(n, mlc.batch_size, gen):
            y = train.labels[idx].to(model.device)
            observed_logits = model.image_logits(train.images[idx], grad=True)

            if mode == "plain_ft":
                loss = F.cross_entropy(observed_logits, y)
            elif mode == "cf_aug":
                cand = model.candidate_logits(train.candidates[idx], grad=True)
                aug = F.cross_entropy(cand.reshape(-1, k), y.repeat_interleave(cand.shape[1]))
                loss = F.cross_entropy(observed_logits, y) + aug
            elif mode == "csa":
                cand = model.candidate_logits(train.candidates[idx], grad=True)
                # clean == observed (no overlay to remove on Waterbirds); stability
                # asks predictions to be invariant to the finite interventions.
                loss, _ = csa_total_loss(
                    observed_logits, y, observed_logits, cand,
                    reference_clean_logits=reference_clean[idx].to(model.device),
                    lambda_stability=cfg.csa.lambda_stability,
                    lambda_cic=cfg.csa.lambda_cic,
                    lambda_preservation=cfg.csa.lambda_preserve,
                    preservation_mode=cfg.csa.preservation_mode,
                )
            elif mode == "group_dro":
                # GROUP-LABEL-SUPERVISED baseline (clearly marked). Uses group ids.
                g = train.groups[idx]
                per_ex = F.cross_entropy(observed_logits, y, reduction="none")
                losses = []
                for gi in range(len(GROUP_NAMES)):
                    sel = g == gi
                    losses.append(per_ex[sel].mean() if sel.any() else observed_logits.new_zeros(()))
                group_loss = torch.stack(losses)
                with torch.no_grad():
                    group_weights = group_weights * torch.exp(cfg.group_dro.eta * group_loss.detach().cpu())
                    group_weights = group_weights / group_weights.sum()
                loss = (group_weights.to(group_loss.device) * group_loss).sum()
            else:
                raise ValueError(f"unknown mode: {mode!r}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# --------------------------------------------------------------------------- #
# Cached-embedding fallback (lightweight head over FROZEN embeddings; NOT LoRA)
# --------------------------------------------------------------------------- #
@dataclass
class WBEmbeddingSplit:
    observed: torch.Tensor    # [N, D]
    candidates: torch.Tensor  # [N, M, D]
    labels: torch.Tensor
    places: torch.Tensor
    groups: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class WBEmbeddingDataset:
    train: WBEmbeddingSplit
    eval_splits: dict[str, WBEmbeddingSplit]
    primary_eval_split: str
    classifier: FrozenClassifier
    embed_dim: int
    num_classes: int
    intervention_summary: dict[str, Any]


def _config_hash(cfg: WBPilotConfig, tag: str) -> str:
    payload = {
        "seed": cfg.seed,
        "data": {k: getattr(cfg.data, k) for k in ("image_size", "max_train_examples", "max_eval_examples", "eval_splits")},
        "iv": cfg.interventions.__dict__,
        "tag": tag,
        "v": 1,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def build_cached_embedding_dataset(cfg: WBPilotConfig, dataset: WaterbirdsDataset, device: str) -> dict[str, Any]:
    """Encode FROZEN OpenCLIP image embeddings for the fallback head.

    Returns ``{"ok": False, "reason": ...}`` if OpenCLIP cannot be loaded.
    """

    from causal_reliability.real_models.clip_zero_shot import (
        check_clip_available,
        encode_images,
        encode_text_prompts,
    )

    status = check_clip_available(
        device=device,
        allow_download=cfg.model.allow_download,
        preferred_backend=cfg.model.backend,
        model_name=cfg.model.model_name,
        pretrained_tag=cfg.model.pretrained_tag,
        transformers_model_name=cfg.model.transformers_model_name,
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return {
            "ok": False,
            "reason": f"real OpenCLIP unavailable ({status.error_message or 'not loaded'})",
            "backend": status.backend,
        }

    bs = cfg.manual_lora.encode_batch_size

    def encode_fn(images_uint8: torch.Tensor) -> torch.Tensor:
        floats = images_uint8.float() / 255.0
        out = []
        for i in range(0, len(floats), bs):
            out.append(encode_images(status, floats[i : i + bs], device).detach().cpu())
        return torch.cat(out, dim=0) if out else torch.empty(0)

    def encode_split(split: WBImageSplit) -> WBEmbeddingSplit:
        obs = encode_fn(split.images)
        n, m = split.candidates.shape[0], split.candidates.shape[1]
        cand_flat = encode_fn(split.candidates.reshape(n * m, *split.candidates.shape[2:]))
        cand = cand_flat.reshape(n, m, -1)
        return WBEmbeddingSplit(obs, cand, split.labels, split.places, split.groups)

    tag = f"{status.backend}:{status.model_name}:{status.pretrained_tag}"
    cache_path = Path(cfg.cache_dir) / f"wb_embeddings_{_config_hash(cfg, tag)}.pt"
    if cfg.use_embedding_cache and cache_path.exists():
        try:
            blob = torch.load(cache_path)
            emb_ds = _unpack_embedding_dataset(blob, dataset, cfg)
            return {"ok": True, "dataset": emb_ds, "backend": status.backend,
                    "model_name": status.model_name, "pretrained_tag": status.pretrained_tag,
                    "from_cache": True}
        except Exception:
            pass

    train_emb = encode_split(dataset.train)
    eval_emb = {name: encode_split(s) for name, s in dataset.eval_splits.items()}
    prompts = [cfg.data.landbird_prompt, cfg.data.waterbird_prompt]
    text_features = encode_text_prompts(status, prompts, device).detach().cpu()
    classifier = FrozenClassifier("clip_text", cfg.model.logit_scale, text_features=text_features)
    emb_ds = WBEmbeddingDataset(
        train=train_emb, eval_splits=eval_emb, primary_eval_split=dataset.primary_eval_split,
        classifier=classifier, embed_dim=int(text_features.shape[1]),
        num_classes=dataset.num_classes, intervention_summary=dataset.intervention_summary,
    )
    if cfg.use_embedding_cache:
        _save_embedding_dataset(cache_path, emb_ds)
    return {"ok": True, "dataset": emb_ds, "backend": status.backend,
            "model_name": status.model_name, "pretrained_tag": status.pretrained_tag,
            "from_cache": False}


def _save_embedding_dataset(path: Path, ds: WBEmbeddingDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def pack(s: WBEmbeddingSplit):
        return {"observed": s.observed, "candidates": s.candidates, "labels": s.labels,
                "places": s.places, "groups": s.groups}

    torch.save({
        "train": pack(ds.train),
        "eval_splits": {k: pack(v) for k, v in ds.eval_splits.items()},
        "primary_eval_split": ds.primary_eval_split,
        "text_features": ds.classifier.text_features,
        "logit_scale": ds.classifier.logit_scale,
        "embed_dim": ds.embed_dim, "num_classes": ds.num_classes,
    }, path)


def _unpack_embedding_dataset(blob: dict, dataset: WaterbirdsDataset, cfg: WBPilotConfig) -> WBEmbeddingDataset:
    def unpack(d):
        return WBEmbeddingSplit(d["observed"], d["candidates"], d["labels"], d["places"], d["groups"])

    classifier = FrozenClassifier("clip_text", float(blob["logit_scale"]), text_features=blob["text_features"])
    return WBEmbeddingDataset(
        train=unpack(blob["train"]),
        eval_splits={k: unpack(v) for k, v in blob["eval_splits"].items()},
        primary_eval_split=str(blob["primary_eval_split"]),
        classifier=classifier, embed_dim=int(blob["embed_dim"]),
        num_classes=int(blob["num_classes"]), intervention_summary=dataset.intervention_summary,
    )


def train_fallback_mode(emb_ds: WBEmbeddingDataset, mode: str, cfg: WBPilotConfig) -> Optional[nn.Module]:
    """Train the lightweight fallback adapter/head over FROZEN embeddings."""

    if mode == "frozen":
        return None
    fb = cfg.fallback
    gen = torch.Generator().manual_seed(cfg.seed + (cfg.training_modes().index(mode) if mode in cfg.training_modes() else 0))
    adapter = EmbeddingAdapter(emb_ds.embed_dim, fb.bottleneck, fb.alpha)
    with torch.no_grad():
        adapter.down.weight.copy_(0.01 * torch.randn(adapter.down.weight.shape, generator=gen))
        adapter.down.bias.zero_()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=fb.lr)
    train = emb_ds.train
    clf = emb_ds.classifier
    k = emb_ds.num_classes
    reference_clean = clf.logits(train.observed).detach() if mode == "csa" else None
    if mode == "group_dro":
        group_weights = torch.ones(len(GROUP_NAMES)) / len(GROUP_NAMES)

    for _ in range(fb.epochs):
        adapter.train()
        for idx in _iter_batches(len(train), fb.batch_size, gen):
            y = train.labels[idx]
            observed_logits = model_logits(adapter, clf, train.observed[idx])
            if mode == "plain_ft":
                loss = F.cross_entropy(observed_logits, y)
            elif mode == "cf_aug":
                cand = model_logits(adapter, clf, train.candidates[idx])
                aug = F.cross_entropy(cand.reshape(-1, k), y.repeat_interleave(cand.shape[1]))
                loss = F.cross_entropy(observed_logits, y) + aug
            elif mode == "csa":
                cand = model_logits(adapter, clf, train.candidates[idx])
                loss, _ = csa_total_loss(
                    observed_logits, y, observed_logits, cand,
                    reference_clean_logits=reference_clean[idx],
                    lambda_stability=cfg.csa.lambda_stability,
                    lambda_cic=cfg.csa.lambda_cic,
                    lambda_preservation=cfg.csa.lambda_preserve,
                    preservation_mode=cfg.csa.preservation_mode,
                )
            elif mode == "group_dro":
                g = train.groups[idx]
                per_ex = F.cross_entropy(observed_logits, y, reduction="none")
                losses = []
                for gi in range(len(GROUP_NAMES)):
                    sel = g == gi
                    losses.append(per_ex[sel].mean() if sel.any() else observed_logits.new_zeros(()))
                group_loss = torch.stack(losses)
                with torch.no_grad():
                    group_weights = group_weights * torch.exp(cfg.group_dro.eta * group_loss.detach())
                    group_weights = group_weights / group_weights.sum()
                loss = (group_weights * group_loss).sum()
            else:
                raise ValueError(f"unknown mode: {mode!r}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    adapter.eval()
    return adapter


# --------------------------------------------------------------------------- #
# Group / worst-group metrics (group labels used ONLY here, for evaluation)
# --------------------------------------------------------------------------- #
def compute_group_metrics(logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> dict[str, Any]:
    """Average accuracy, worst-group accuracy, and per-group accuracies.

    ``worst_group_accuracy`` is the minimum over groups that actually have
    examples. Groups with no examples report ``None`` and are excluded from the
    worst-group minimum.
    """

    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float()
    average_accuracy = float(correct.mean()) if len(correct) else 0.0

    group_acc: dict[str, Optional[float]] = {}
    group_counts: dict[str, int] = {}
    present_accs: list[float] = []
    for gi, name in enumerate(GROUP_NAMES):
        sel = groups == gi
        cnt = int(sel.sum())
        group_counts[name] = cnt
        if cnt == 0:
            group_acc[name] = None
            continue
        acc = float(correct[sel].mean())
        group_acc[name] = acc
        present_accs.append(acc)
    worst = min(present_accs) if present_accs else 0.0
    return {
        "average_accuracy": average_accuracy,
        "worst_group_accuracy": worst,
        "group_accuracies": group_acc,
        "group_counts": group_counts,
    }


@torch.no_grad()
def evaluate_manual_lora(model: ManualLoraModel, split: WBImageSplit) -> dict[str, Any]:
    logits = model.image_logits(split.images, classifier="shape", grad=False).cpu()
    cand_logits = model.candidate_logits(split.candidates, classifier="shape", grad=False).cpu()
    out = compute_group_metrics(logits, split.labels, split.groups)
    out["counterfactual_instability"] = float(cic_instability_penalty(cand_logits))
    return out


@torch.no_grad()
def evaluate_fallback(adapter: Optional[nn.Module], emb_ds: WBEmbeddingDataset, split: WBEmbeddingSplit) -> dict[str, Any]:
    clf = emb_ds.classifier
    logits = model_logits(adapter, clf, split.observed)
    cand_logits = model_logits(adapter, clf, split.candidates)
    out = compute_group_metrics(logits, split.labels, split.groups)
    out["counterfactual_instability"] = float(cic_instability_penalty(cand_logits))
    return out


# --------------------------------------------------------------------------- #
# Go / no-go (pre-registered)
# --------------------------------------------------------------------------- #
def compute_go_no_go(
    plain: dict[str, Any],
    csa: dict[str, Any],
    *,
    real_lora_used: bool,
    cfg: WBPilotConfig,
) -> dict[str, Any]:
    """Single-seed pre-registered go / no-go on the primary eval split.

    ``waterbirds_csa_promising`` requires real ``manual_lora_visual`` (never the
    cached fallback), plus: avg-accuracy drop vs plain <= 0.03, worst-group gain
    vs plain >= +0.05, and >= 20% relative CIC-instability reduction vs plain.
    """

    avg_drop_vs_plain = float(plain["average_accuracy"] - csa["average_accuracy"])
    worst_group_gain_vs_plain = float(csa["worst_group_accuracy"] - plain["worst_group_accuracy"])
    instab_plain = float(plain["counterfactual_instability"])
    instab_csa = float(csa["counterfactual_instability"])
    instability_drop_rel = (instab_plain - instab_csa) / instab_plain if instab_plain > 1e-9 else 0.0

    avg_drop_ok = avg_drop_vs_plain <= cfg.max_avg_acc_drop
    worst_group_gain_ok = worst_group_gain_vs_plain >= cfg.min_worst_group_gain
    instability_drop_ok = instability_drop_rel >= cfg.min_instability_drop_rel

    promising = bool(real_lora_used and avg_drop_ok and worst_group_gain_ok and instability_drop_ok)

    # null: any of the pre-registered failure conditions.
    instability_improves_but_robustness_does_not = bool(instability_drop_ok and not worst_group_gain_ok)
    null = bool(
        (not real_lora_used)
        or (worst_group_gain_vs_plain < cfg.min_worst_group_gain)
        or (avg_drop_vs_plain > cfg.max_avg_acc_drop)
        or instability_improves_but_robustness_does_not
    )

    return {
        "thresholds": {
            "max_avg_acc_drop": cfg.max_avg_acc_drop,
            "min_worst_group_gain": cfg.min_worst_group_gain,
            "min_instability_drop_rel": cfg.min_instability_drop_rel,
            "strong_worst_group_gain": cfg.strong_worst_group_gain,
        },
        "real_lora_used": bool(real_lora_used),
        "avg_acc_drop_vs_plain": avg_drop_vs_plain,
        "worst_group_gain_vs_plain": worst_group_gain_vs_plain,
        "instability_drop_rel_vs_plain": instability_drop_rel,
        "avg_drop_ok": bool(avg_drop_ok),
        "worst_group_gain_ok": bool(worst_group_gain_ok),
        "instability_drop_ok": bool(instability_drop_ok),
        "instability_improves_but_robustness_does_not": instability_improves_but_robustness_does_not,
        "waterbirds_csa_promising": promising,
        "waterbirds_csa_null": null,
    }


def compute_strong_flag(per_seed: list[dict[str, Any]], *, real_lora_used: bool, cfg: WBPilotConfig) -> dict[str, Any]:
    """Multi-seed (0,1,2) pre-registered ``waterbirds_csa_strong`` flag.

    ``per_seed`` is a list of ``{"seed", "plain", "csa"}`` dicts (primary-split
    metrics). Strong requires seeds {0,1,2} complete with real LoRA, mean paired
    worst-group gain >= +0.08 exceeding the seed-to-seed std of paired gains, and
    mean avg-accuracy drop <= 0.03.
    """

    seeds = sorted(int(r["seed"]) for r in per_seed)
    have_three = real_lora_used and set(seeds) >= {0, 1, 2}
    if not have_three:
        return {
            "waterbirds_csa_strong": False,
            "reason": ("requires real manual_lora_visual and seeds {0,1,2}; "
                       f"have real_lora_used={real_lora_used}, seeds={seeds}"),
            "seeds": seeds,
        }

    import statistics

    by_seed = {int(r["seed"]): r for r in per_seed}
    diffs = [by_seed[s]["csa"]["worst_group_accuracy"] - by_seed[s]["plain"]["worst_group_accuracy"] for s in (0, 1, 2)]
    avg_drops = [by_seed[s]["plain"]["average_accuracy"] - by_seed[s]["csa"]["average_accuracy"] for s in (0, 1, 2)]
    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.pstdev(diffs)
    mean_avg_drop = statistics.fmean(avg_drops)

    beats_threshold = mean_diff >= cfg.strong_worst_group_gain
    exceeds_std = mean_diff > std_diff
    avg_ok = mean_avg_drop <= cfg.max_avg_acc_drop
    strong = bool(beats_threshold and exceeds_std and avg_ok)
    return {
        "waterbirds_csa_strong": strong,
        "seeds": seeds,
        "mean_paired_worst_group_gain": mean_diff,
        "paired_worst_group_gain_std": std_diff,
        "mean_avg_acc_drop": mean_avg_drop,
        "beats_strong_threshold": bool(beats_threshold),
        "gain_exceeds_seed_std": bool(exceeds_std),
        "avg_drop_ok": bool(avg_ok),
    }


__all__ = [
    "WATERBIRD_CLASSES",
    "GROUP_NAMES",
    "BASE_MODES",
    "OPTIONAL_MODES",
    "INTERVENTION_TYPES",
    "group_index",
    "WBModelConfig",
    "WBDataConfig",
    "WBInterventionConfig",
    "WBManualLoraConfig",
    "WBCsaConfig",
    "WBFallbackConfig",
    "WBGroupDROConfig",
    "WBPilotConfig",
    "detect_device",
    "is_accelerator",
    "decide_mode",
    "waterbirds_dataset_dir",
    "check_waterbirds_available",
    "WBImageSplit",
    "WaterbirdsDataset",
    "load_waterbirds_dataset",
    "describe_interventions",
    "build_candidate_interventions",
    "build_real_lora_model",
    "train_manual_lora_mode",
    "evaluate_manual_lora",
    "WBEmbeddingSplit",
    "WBEmbeddingDataset",
    "build_cached_embedding_dataset",
    "train_fallback_mode",
    "evaluate_fallback",
    "compute_group_metrics",
    "compute_go_no_go",
    "compute_strong_flag",
]
