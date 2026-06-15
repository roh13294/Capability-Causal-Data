"""Bounded Waterbirds pilot: finite-candidate CIC background-shortcut repair.

This is OPTIONAL supporting evidence for CIC, not a replacement for the main
OpenCLIP text-overlay result. Waterbirds is a real spurious-background
benchmark: the causal label is bird type (landbird/waterbird) and the shortcut
is background habitat (land/water). The pilot tests whether a *finite-candidate*
CIC intervention can identify/neutralize the spurious background shortcut when an
oracle-repairable bird/background mask is available.

Hard scope guarantees enforced here:
* Skips cleanly (never errors, never fabricates) if the dataset or
  oracle-repairable masks/bboxes are unavailable.
* Never claims open-world discovery: the candidate set is finite and explicit.
* Never claims general robustness.
* Non-oracle CIC scoring uses ONLY model predictions -- no true label, group
  label, or test-correctness leakage.
* Headline eligibility is gated (PART 6). A failed gate writes a pilot/negative
  record; it does not become a main positive result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageFilter

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.real_models.clip_zero_shot import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
    DEFAULT_TRANSFORMERS_MODEL,
    ClipStatus,
    ClipZeroShotClassifier,
    check_clip_available,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


CLASS_NAMES = ["landbird", "waterbird"]
SKIP_NO_MASK_MESSAGE = "Waterbirds pilot skipped: no oracle-repairable bird/background mask available."
SKIP_NO_DATA_MESSAGE = "Waterbirds pilot skipped: no Waterbirds-style dataset found locally."
WILDS_NO_MASK_SUMMARY = (
    "WILDS Waterbirds was found and parsed, but no oracle-repairable masks/bboxes "
    "were available, so the repair pilot is not headline-eligible."
)

# Label / background code -> human-readable name (WILDS Waterbirds convention).
LABEL_NAMES = {0: "landbird", 1: "waterbird"}
BACKGROUND_NAMES = {0: "land", 1: "water"}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}

# CIC ranks among these finite candidate interventions. This is the entire
# candidate space -- there is no open-world search.
CIC_CANDIDATE_METHODS = [
    "oracle_background_grayfill",
    "oracle_background_blur",
    "replace_background_neutral",
    "crop_around_bird",
    "preserve_bird_region",
    "random_background_patch_blur",
    "bird_region_blur",
    "largest_region_neutralization",
    "center_crop",
]


# ---------------------------------------------------------------------------
# device / download helpers
# ---------------------------------------------------------------------------
def _device(model_cfg: dict[str, Any], cfg: dict[str, Any]) -> str:
    requested = str(model_cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() and bool(cfg.get("prefer_gpu", False)) else "cpu"
    return requested


def _downloads_allowed(model_cfg: dict[str, Any]) -> bool:
    return bool(model_cfg.get("allow_pretrained_download", False))


def _pil_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def _predict_pil(model: ClipZeroShotClassifier, images: list[Image.Image]) -> np.ndarray:
    return np.asarray(model.predict(_pil_to_tensor(images))["probabilities"].detach().cpu().numpy(), dtype=np.float64)


# ---------------------------------------------------------------------------
# dataset availability gate (PART 1)
# ---------------------------------------------------------------------------
def _resolve(root: Path, value: Any, sub: Path | None = None) -> Path | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    if sub is not None:
        cand = sub / p
        if cand.exists():
            return cand
    return root / p


def _has_bbox(row: pd.Series, bbox_cols: list[str]) -> bool:
    if not all(c in row.index for c in bbox_cols):
        return False
    try:
        vals = [float(row[c]) for c in bbox_cols]
    except (TypeError, ValueError):
        return False
    if any(math.isnan(v) for v in vals):
        return False
    x0, y0, x1, y1 = vals
    return x1 > x0 and y1 > y0


def _wilds_data_dir(wilds_root: Any, dataset_name: str, download: bool = False) -> Path | None:
    """Resolve the WILDS dataset release directory (e.g. data/wilds/waterbirds_v1.0).

    Robust direct detection is preferred (no heavy import). If the release dir is
    not on disk, optionally fall back to the WILDS loader, which is also the only
    path that may download when ``download=True``.
    """
    root = Path(str(wilds_root))
    candidates = sorted(root.glob(f"{dataset_name}_v*"))
    for cand in candidates:
        if (cand / "metadata.csv").exists():
            return cand
    try:  # pragma: no cover - exercised only when wilds is installed
        from wilds import get_dataset

        dataset = get_dataset(dataset=dataset_name, download=download, root_dir=str(root))
        data_dir = Path(getattr(dataset, "_data_dir", getattr(dataset, "data_dir", root)))
        if (data_dir / "metadata.csv").exists():
            return data_dir
    except Exception:
        pass
    return None


def _scan_dir_for_masks(root: Path, limit: int = 20000) -> list[str]:
    """Search the dataset tree for files whose names suggest masks/segmentations/boxes."""
    found: list[str] = []
    try:
        for p in root.rglob("*"):
            name = p.name.lower()
            if any(tok in name for tok in ("mask", "seg", "bbox")):
                found.append(str(p))
                if len(found) >= 64:
                    break
            limit -= 1
            if limit <= 0:
                break
    except Exception:  # pragma: no cover - defensive
        pass
    return found


def check_dataset(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a structured availability report. Never raises on missing data."""
    source = str(data_cfg.get("source", "local") or "local").lower()
    report: dict[str, Any] = {
        "dataset_available": False,
        "masks_available": False,
        "oracle_repair_available": False,
        "n_rows": 0,
        "source": source,
        "root": "",
        "metadata_csv": "",
        "reason": "",
        "uses_mask": False,
        "uses_bbox": False,
        "mask_files_in_dir": 0,
    }

    if source == "wilds":
        data_dir = _wilds_data_dir(
            data_cfg.get("wilds_root", "data/wilds"),
            str(data_cfg.get("wilds_dataset", "waterbirds")),
            download=bool(data_cfg.get("download", False)),
        )
        if data_dir is None:
            report["reason"] = (
                f"{SKIP_NO_DATA_MESSAGE} (WILDS {data_cfg.get('wilds_dataset', 'waterbirds')} "
                f"not found under {data_cfg.get('wilds_root', 'data/wilds')})"
            )
            return report
        root = data_dir
        metadata_csv = data_dir / "metadata.csv"
    else:
        report["source"] = "local_directory"
        root = Path(str(data_cfg.get("root", data_cfg.get("local_root", "data/waterbirds"))))
        meta_cfg = data_cfg.get("metadata_csv")
        metadata_csv = Path(str(meta_cfg)) if meta_cfg else root / "metadata.csv"

    report["root"] = str(root)
    report["metadata_csv"] = str(metadata_csv)
    if not metadata_csv.exists():
        report["reason"] = f"{SKIP_NO_DATA_MESSAGE} (no metadata at {metadata_csv})"
        return report
    try:
        df = pd.read_csv(metadata_csv)
    except Exception as exc:  # pragma: no cover - defensive
        report["reason"] = f"{SKIP_NO_DATA_MESSAGE} (unreadable metadata: {exc})"
        return report

    image_col = str(data_cfg.get("image_column", "img_filename"))
    label_col = str(data_cfg.get("label_column", "y"))
    if image_col not in df.columns or label_col not in df.columns:
        report["reason"] = f"{SKIP_NO_DATA_MESSAGE} (missing '{image_col}' or '{label_col}' column)"
        return report

    # At least one image file must actually exist on disk.
    image_dir = _resolve(root, data_cfg.get("image_dir")) or root
    sample = df.head(64)
    images_present = any((_resolve(root, v, image_dir) or root).exists() for v in sample[image_col].tolist())
    if not images_present:
        report["reason"] = f"{SKIP_NO_DATA_MESSAGE} (no image files found under {root})"
        return report

    report["dataset_available"] = True
    report["n_rows"] = int(len(df))

    # Oracle-repairable isolation gate: a usable mask OR a valid bird bbox.
    mask_col = data_cfg.get("mask_column", "mask_filename")
    bbox_cols = list(data_cfg.get("bbox_columns", ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]) or [])
    mask_dir = _resolve(root, data_cfg.get("mask_dir")) or root
    masks_present = False
    if mask_col and str(mask_col) in df.columns:
        masks_present = any(
            (_resolve(root, v, mask_dir) or root).exists()
            for v in sample[str(mask_col)].tolist()
            if isinstance(v, str) or (v is not None and not (isinstance(v, float) and math.isnan(v)))
        )
        report["uses_mask"] = masks_present
    bbox_present = bool(bbox_cols) and any(_has_bbox(row, bbox_cols) for _, row in sample.iterrows())
    report["uses_bbox"] = bbox_present and not masks_present

    # PART 3: scan the dataset tree for mask/seg/bbox files (informational). Their
    # mere presence does NOT grant oracle availability -- only per-example masks or
    # valid bboxes that isolate the bird do, to preserve the no-leakage guarantee.
    report["mask_files_in_dir"] = len(_scan_dir_for_masks(root))

    if not (masks_present or bbox_present):
        report["reason"] = SKIP_NO_MASK_MESSAGE
        return report

    report["masks_available"] = True
    # Oracle repair requires masks/bboxes; require_masks_for_oracle stays honored.
    report["oracle_repair_available"] = bool(masks_present or bbox_present)
    report["reason"] = "available"
    return report


def _load_examples(data_cfg: dict[str, Any], avail: dict[str, Any], size: int, seed: int) -> list[dict[str, Any]]:
    root = Path(str(avail.get("root") or data_cfg.get("root", "data/waterbirds")))
    metadata_csv = Path(str(avail.get("metadata_csv") or (root / "metadata.csv")))
    df = pd.read_csv(metadata_csv)
    image_col = str(data_cfg.get("image_column", "img_filename"))
    label_col = str(data_cfg.get("label_column", "y"))
    place_col = str(data_cfg.get("place_column", "place"))
    split_col = str(data_cfg.get("split_column", "split"))
    mask_col = data_cfg.get("mask_column", "mask_filename")
    bbox_cols = list(data_cfg.get("bbox_columns", ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]) or [])
    image_dir = _resolve(root, data_cfg.get("image_dir")) or root
    mask_dir = _resolve(root, data_cfg.get("mask_dir")) or root

    eval_split = data_cfg.get("eval_split", "test")
    split_map = {"train": 0, "val": 1, "validation": 1, "test": 2}
    if split_col in df.columns and eval_split is not None:
        want = split_map.get(str(eval_split).lower())
        if want is not None and (df[split_col] == want).any():
            df = df[df[split_col] == want]
    df = df.reset_index(drop=True)

    max_n = int(data_cfg.get("max_natural_examples", 200))
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=seed).reset_index(drop=True)

    examples: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        img_path = _resolve(root, row[image_col], image_dir)
        if img_path is None or not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        except Exception:  # pragma: no cover - defensive
            continue
        mask = None
        if mask_col and str(mask_col) in df.columns:
            mpath = _resolve(root, row[str(mask_col)], mask_dir)
            if mpath is not None and mpath.exists():
                try:
                    marr = np.asarray(Image.open(mpath).convert("L").resize((size, size), Image.Resampling.NEAREST))
                    mask = marr > 127
                except Exception:  # pragma: no cover - defensive
                    mask = None
        if mask is None and bbox_cols and _has_bbox(row, bbox_cols):
            mask = _bbox_to_mask(row, bbox_cols, size)
        if mask is None or not mask.any() or mask.all():
            # No usable bird/background separation for this example.
            continue
        label = int(row[label_col])
        if label not in (0, 1):
            continue
        examples.append(
            {
                "example_id": int(idx),
                "image": img,
                "mask": mask,
                "label": label,
                "place": (int(row[place_col]) if place_col in df.columns and not pd.isna(row[place_col]) else -1),
                "split": str(eval_split),
            }
        )
    return examples


def _bbox_to_mask(row: pd.Series, bbox_cols: list[str], size: int) -> np.ndarray | None:
    try:
        x0, y0, x1, y1 = (float(row[c]) for c in bbox_cols)
    except (TypeError, ValueError):
        return None
    # bbox is assumed in resized-image coordinates if <= size, else normalized.
    if max(x1, y1) <= 1.5:
        x0, y0, x1, y1 = x0 * size, y0 * size, x1 * size, y1 * size
    mask = np.zeros((size, size), dtype=bool)
    xa, ya = max(0, int(x0)), max(0, int(y0))
    xb, yb = min(size, int(round(x1))), min(size, int(round(y1)))
    if xb <= xa or yb <= ya:
        return None
    mask[ya:yb, xa:xb] = True
    return mask


# ---------------------------------------------------------------------------
# finite candidate interventions (PART 3)
# ---------------------------------------------------------------------------
def _blur_radius(size: int, frac: float) -> float:
    return max(1.5, float(frac) * size)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _neutralize_background(img: Image.Image, mask: np.ndarray, mode: str, *, fill: float, radius: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    bg = ~mask
    out = arr.copy()
    if mode == "gray":
        out[bg] = float(fill)
    elif mode == "neutral_mean":
        bird_mean = arr[mask].mean(axis=0) if mask.any() else np.array([fill, fill, fill])
        out[bg] = bird_mean
    elif mode == "blur":
        blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGB")).astype(np.float32) / 255.0
        out[bg] = blurred[bg]
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))


def _preserve_bird_region(img: Image.Image, mask: np.ndarray, fill: float) -> Image.Image:
    # Keep bird pixels, blank everything else to neutral gray (maximal removal).
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    out = np.full_like(arr, float(fill))
    out[mask] = arr[mask]
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))


def _crop_around_bird(img: Image.Image, mask: np.ndarray, pad_frac: float) -> Image.Image:
    size = img.size[0]
    x0, y0, x1, y1 = _mask_bbox(mask)
    pad = int(pad_frac * size)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(size, x1 + pad), min(size, y1 + pad)
    return img.crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.BICUBIC)


def _center_crop(img: Image.Image, frac: float = 0.7) -> Image.Image:
    size = img.size[0]
    side = int(size * frac)
    off = (size - side) // 2
    return img.crop((off, off, off + side, off + side)).resize((size, size), Image.Resampling.BICUBIC)


def _blur_bbox(img: Image.Image, bbox: tuple[int, int, int, int], radius: float) -> Image.Image:
    out = img.copy()
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return out
    patch = out.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=radius))
    out.paste(patch, (x0, y0))
    return out


def _random_background_patch(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    size = mask.shape[0]
    bg_frac = float((~mask).mean())
    target = float(np.clip(bg_frac, 0.10, 0.6))
    side = int(round(size * math.sqrt(target)))
    side = int(np.clip(side, max(4, size // 8), size))
    x0 = int(rng.integers(0, max(1, size - side + 1)))
    y0 = int(rng.integers(0, max(1, size - side + 1)))
    return x0, y0, x0 + side, y0 + side


def _largest_region_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    # Largest "region" candidate = whole-background bounding box (control: too big).
    size = mask.shape[0]
    return 0, 0, size, size


def build_interventions(ex: dict[str, Any], cfg_iv: dict[str, Any], rng: np.random.Generator) -> dict[str, Image.Image]:
    img = ex["image"]
    mask = ex["mask"]
    size = img.size[0]
    fill = float(cfg_iv.get("background_fill_gray", 0.5))
    radius = _blur_radius(size, float(cfg_iv.get("blur_radius_frac", 0.045)))
    pad_frac = float(cfg_iv.get("crop_pad_frac", 0.10))
    bird_bbox = _mask_bbox(mask)
    return {
        "no_intervention": img,
        # Oracle / mask-based background neutralizations (preserve the bird):
        "oracle_background_grayfill": _neutralize_background(img, mask, "gray", fill=fill, radius=radius),
        "oracle_background_blur": _neutralize_background(img, mask, "blur", fill=fill, radius=radius),
        "replace_background_neutral": _neutralize_background(img, mask, "neutral_mean", fill=fill, radius=radius),
        "crop_around_bird": _crop_around_bird(img, mask, pad_frac),
        "preserve_bird_region": _preserve_bird_region(img, mask, fill),
        # Controls:
        "random_background_patch_blur": _blur_bbox(img, _random_background_patch(mask, rng), radius),
        "bird_region_blur": _blur_bbox(img, bird_bbox, radius),
        "largest_region_neutralization": _blur_bbox(img, _largest_region_bbox(mask), radius),
        "center_crop": _center_crop(img),
    }


# ---------------------------------------------------------------------------
# non-oracle CIC scoring (PART 3) -- no label/group/correctness used
# ---------------------------------------------------------------------------
def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def cic_rank(
    candidate_probs: dict[str, np.ndarray],
    original_probs: np.ndarray,
) -> list[tuple[str, float]]:
    """Rank candidate interventions by counterfactual instability.

    Uses ONLY the model's predicted distributions: the change a candidate
    induces (total variation from the original distribution) tempered by the
    post-intervention confidence (to down-weight interventions that merely
    collapse the prediction, e.g. blurring the bird itself). No true label,
    group label, or correctness signal is consulted.
    """
    ranked: list[tuple[str, float]] = []
    for name, probs in candidate_probs.items():
        instability = _total_variation(original_probs, probs)
        confidence_after = float(np.max(probs))
        score = instability * (0.5 + 0.5 * confidence_after)
        ranked.append((name, score))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# evaluation (PART 4)
# ---------------------------------------------------------------------------
def _evaluate(
    examples: list[dict[str, Any]],
    model: ClipZeroShotClassifier,
    prompts: list[str],
    cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cfg_iv = cfg.get("intervention", {})
    top_k = int(cfg_iv.get("top_k", 3))
    rows: list[dict[str, Any]] = []
    for ex in examples:
        interventions = build_interventions(ex, cfg_iv, rng)
        names = list(interventions.keys())
        probs_arr = _predict_pil(model, [interventions[n] for n in names])
        probs = {n: probs_arr[i] for i, n in enumerate(names)}
        original_probs = probs["no_intervention"]

        # CIC ranks the finite candidate set (excludes the null no-op itself).
        candidate_probs = {n: probs[n] for n in CIC_CANDIDATE_METHODS if n in probs}
        ranked = cic_rank(candidate_probs, original_probs)
        cic_top1_name = ranked[0][0] if ranked else "no_intervention"
        cic_topk_names = [n for n, _ in ranked[:top_k]]
        # top-k consensus vote on predicted class.
        topk_preds = [int(np.argmax(probs[n])) for n in cic_topk_names]
        if topk_preds:
            vals, counts = np.unique(topk_preds, return_counts=True)
            cic_topk_pred = int(vals[int(counts.argmax())])
        else:
            cic_topk_pred = int(np.argmax(original_probs))

        label = ex["label"]
        row: dict[str, Any] = {
            "example_id": ex["example_id"],
            "label": label,
            "place": ex["place"],
            "cic_top1_selected": cic_top1_name,
            "cic_topk_selected": "|".join(cic_topk_names),
        }
        for name in names:
            pred = int(np.argmax(probs[name]))
            row[f"{name}__pred"] = pred
            row[f"{name}__conf"] = float(np.max(probs[name]))
            row[f"{name}__correct"] = bool(pred == label)
        # Derived CIC methods.
        row["cic_top1__pred"] = int(np.argmax(probs[cic_top1_name]))
        row["cic_top1__correct"] = bool(int(np.argmax(probs[cic_top1_name])) == label)
        row["cic_topk_consensus__pred"] = cic_topk_pred
        row["cic_topk_consensus__correct"] = bool(cic_topk_pred == label)
        rows.append(row)
    return pd.DataFrame(rows)


REPORT_METHODS = [
    "no_intervention",
    "oracle_background_grayfill",
    "oracle_background_blur",
    "replace_background_neutral",
    "crop_around_bird",
    "preserve_bird_region",
    "cic_top1",
    "cic_topk_consensus",
    "random_background_patch_blur",
    "bird_region_blur",
    "largest_region_neutralization",
    "center_crop",
]


def _accuracy(certs: pd.DataFrame, method: str) -> tuple[float, int]:
    col = f"{method}__correct"
    if col not in certs.columns or len(certs) == 0:
        return float("nan"), 0
    vals = certs[col].astype(bool)
    return float(vals.mean()), int(len(vals))


def _wald_halfwidth(acc: float, n: int) -> float:
    if n <= 0 or not np.isfinite(acc):
        return float("nan")
    return float(1.96 * math.sqrt(max(acc * (1 - acc), 1e-9) / n))


def _failure_conditioned(certs: pd.DataFrame) -> pd.DataFrame:
    """Mode B: original wrong, oracle restores correct, bird still recognizable."""
    if len(certs) == 0:
        return certs
    cond = (
        (~certs["no_intervention__correct"].astype(bool))
        & certs["oracle_background_grayfill__correct"].astype(bool)
    )
    if "crop_around_bird__correct" in certs.columns:
        cond = cond & certs["crop_around_bird__correct"].astype(bool)
    return certs[cond].reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics + eligibility (PART 5 / PART 6)
# ---------------------------------------------------------------------------
def _metric_rows(
    certs_nat: pd.DataFrame,
    certs_fc: pd.DataFrame,
    status: ClipStatus,
    headline: bool,
    reasons: list[str],
    avail: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evidence = "pretrained CLIP finite-candidate background-shortcut pilot" if status.pretrained else "unavailable"
    for method in REPORT_METHODS:
        nat_acc, nat_n = _accuracy(certs_nat, method)
        fc_acc, fc_n = _accuracy(certs_fc, method)
        rows.append(
            {
                "method": method,
                "evidence_status": evidence,
                "headline_eligible": bool(headline),
                "include_in_final_headline": bool(headline),
                "headline_eligibility_reasons": "eligible" if headline else "; ".join(reasons),
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": bool(status.pretrained),
                "dataset_available": bool(avail.get("dataset_available", False)),
                "masks_available": bool(avail.get("masks_available", False)),
                "finite_candidate_not_open_world": True,
                "natural_n": nat_n,
                "natural_accuracy": nat_acc,
                "natural_accuracy_ci95_halfwidth": _wald_halfwidth(nat_acc, nat_n),
                "failure_conditioned_n": fc_n,
                "failure_conditioned_accuracy": fc_acc,
                "failure_conditioned_accuracy_ci95_halfwidth": _wald_halfwidth(fc_acc, fc_n),
            }
        )
    return pd.DataFrame(rows)


def _headline_eligibility(
    certs_nat: pd.DataFrame,
    certs_fc: pd.DataFrame,
    status: ClipStatus,
    avail: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    elig = cfg.get("eligibility", {})
    if status.backend not in {"open_clip", "transformers"} or not status.pretrained:
        reasons.append("pretrained CLIP backend did not load")
    if status.backend == "fake":
        reasons.append("fake backend cannot be headline eligible")
    if not avail.get("dataset_available", False):
        reasons.append("dataset unavailable")
    if not avail.get("masks_available", False):
        reasons.append("oracle-repairable masks/bboxes unavailable")
    if len(certs_nat) == 0:
        reasons.append("no evaluated examples")
        return False, reasons

    orig_acc, _ = _accuracy(certs_nat, "no_intervention")
    oracle_acc, _ = _accuracy(certs_nat, "oracle_background_grayfill")
    cic1_acc, _ = _accuracy(certs_nat, "cic_top1")
    cick_acc, _ = _accuracy(certs_nat, "cic_topk_consensus")
    rand_acc, rand_n = _accuracy(certs_nat, "random_background_patch_blur")
    bird_pres_acc, _ = _accuracy(certs_nat, "crop_around_bird")
    n_fc = int(len(certs_fc))

    min_fc = int(elig.get("min_failure_conditioned", 30))
    low_thr = float(elig.get("low_natural_accuracy_threshold", 0.65))
    if not (n_fc >= min_fc or (np.isfinite(orig_acc) and orig_acc <= low_thr)):
        reasons.append(
            f"insufficient real shortcut failures (failure-conditioned n={n_fc} < {min_fc} and natural orig acc {orig_acc:.3f} > {low_thr})"
        )
    if not (np.isfinite(oracle_acc) and oracle_acc >= float(elig.get("oracle_min_accuracy", 0.80))):
        reasons.append(f"oracle repair accuracy {oracle_acc:.3f} < {elig.get('oracle_min_accuracy', 0.80)}")

    best_cic = max([a for a in [cic1_acc, cick_acc] if np.isfinite(a)] or [float("nan")])
    margin_req = float(elig.get("cic_beats_random_abs", 0.15))
    beats_abs = np.isfinite(best_cic) and np.isfinite(rand_acc) and (best_cic - rand_acc) >= margin_req
    # CI non-overlap alternative.
    ci_nonoverlap = False
    if np.isfinite(best_cic) and np.isfinite(rand_acc):
        cic_n = len(certs_nat)
        lo_cic = best_cic - _wald_halfwidth(best_cic, cic_n)
        hi_rand = rand_acc + _wald_halfwidth(rand_acc, rand_n)
        ci_nonoverlap = lo_cic > hi_rand
    if not (beats_abs or ci_nonoverlap):
        reasons.append(
            f"CIC does not beat matched random by >= {margin_req} (best CIC {best_cic:.3f} vs random {rand_acc:.3f})"
        )

    min_pres = float(elig.get("min_bird_preservation_accuracy", 0.70))
    if not (np.isfinite(bird_pres_acc) and bird_pres_acc >= min_pres):
        reasons.append(f"bird preservation proxy {bird_pres_acc:.3f} < {min_pres}")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# output writers (PART 5)
# ---------------------------------------------------------------------------
def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    plt.figure(figsize=(9.6, 4.8))
    plottable = metrics[metrics.get("natural_accuracy", pd.Series(dtype=float)).notna()] if len(metrics) else pd.DataFrame()
    if len(plottable):
        x = np.arange(len(plottable))
        colors = ["#4c78a8" if "cic" in m else ("#54a24b" if "oracle" in m else "#bab0ac") for m in plottable["method"]]
        plt.bar(x, plottable["natural_accuracy"], color=colors)
        if "natural_accuracy_ci95_halfwidth" in plottable:
            plt.errorbar(x, plottable["natural_accuracy"], yerr=plottable["natural_accuracy_ci95_halfwidth"], fmt="none", ecolor="#333333", capsize=3)
        plt.xticks(x, plottable["method"], rotation=30, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("Natural held-out accuracy")
        plt.title("Waterbirds finite-candidate CIC pilot")
    else:
        plt.text(0.5, 0.5, "Waterbirds pilot skipped (no data / masks / model)", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _write_outputs(
    out_dir: Path,
    cfg: dict[str, Any],
    metrics: pd.DataFrame,
    certs_nat: pd.DataFrame,
    certs_fc: pd.DataFrame,
    key_numbers: dict[str, Any],
    summary_lines: list[str],
    examples_md: str,
    caption_md: str,
) -> dict[str, str]:
    metrics.to_csv(out_dir / "waterbirds_metrics.csv", index=False)
    certs_nat.to_csv(out_dir / "waterbirds_certificates.csv", index=False)
    certs_fc.to_csv(out_dir / "waterbirds_failure_conditioned_certificates.csv", index=False)
    (out_dir / "waterbirds_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "waterbirds_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    (out_dir / "waterbirds_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (out_dir / "waterbirds_examples.md").write_text(examples_md, encoding="utf-8")
    (out_dir / "waterbirds_caption.md").write_text(caption_md, encoding="utf-8")
    _plot(metrics, out_dir / "waterbirds_plot.png", out_dir / "waterbirds_plot.pdf")
    return {
        "metrics": str(out_dir / "waterbirds_metrics.csv"),
        "certificates": str(out_dir / "waterbirds_certificates.csv"),
        "key_numbers": str(out_dir / "waterbirds_key_numbers.json"),
        "summary": str(out_dir / "waterbirds_summary.md"),
        "examples": str(out_dir / "waterbirds_examples.md"),
        "caption": str(out_dir / "waterbirds_caption.md"),
        "plot": str(out_dir / "waterbirds_plot.png"),
    }


def _write_skipped(
    out_dir: Path,
    cfg: dict[str, Any],
    avail: dict[str, Any],
    status: ClipStatus,
    reason: str,
    extra_outputs: dict[str, str] | None = None,
) -> dict[str, str]:
    extra_outputs = extra_outputs or {}
    dataset_available = bool(avail.get("dataset_available", False))
    masks_available = bool(avail.get("masks_available", False))
    oracle_available = bool(avail.get("oracle_repair_available", masks_available))
    is_wilds = str(avail.get("source", "")) == "wilds"
    wilds_parsed = is_wilds and dataset_available
    metrics = pd.DataFrame(
        [
            {
                "method": "skipped",
                "evidence_status": "unavailable",
                "headline_eligible": False,
                "include_in_final_headline": False,
                "headline_eligibility_reasons": reason,
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": bool(status.pretrained),
                "dataset_available": dataset_available,
                "masks_available": masks_available,
                "oracle_repair_available": oracle_available,
                "finite_candidate_not_open_world": True,
                "natural_n": 0,
                "natural_accuracy": float("nan"),
                "failure_conditioned_n": 0,
                "failure_conditioned_accuracy": float("nan"),
            }
        ]
    )
    key_numbers = {
        "dataset_available": dataset_available,
        "masks_available": masks_available,
        "oracle_repair_available": oracle_available,
        "cic_repair_ran": False,
        "source": avail.get("source", ""),
        "n_parsed": int(avail.get("n_rows", 0)),
        "mask_files_in_dir": int(avail.get("mask_files_in_dir", 0)),
        "n_natural": 0,
        "n_failure_conditioned": 0,
        "original_accuracy": None,
        "oracle_repair_accuracy": None,
        "cic_top1_accuracy": None,
        "cic_topk_accuracy": None,
        "matched_random_accuracy": None,
        "bird_preservation_accuracy": None,
        "nonoracle_scorer_excluded_label_group_correctness": True,
        "masks_used_for": "n/a (skipped)",
        "waterbirds_headline_eligible": False,
        "skipped": True,
        "skip_reason": reason,
        "finite_candidate_not_open_world": True,
    }
    if "wilds_converted_metadata" in extra_outputs:
        key_numbers["wilds_converted_metadata"] = extra_outputs["wilds_converted_metadata"]
    if "wilds_metadata_diagnostic" in extra_outputs:
        key_numbers["wilds_metadata_diagnostic"] = extra_outputs["wilds_metadata_diagnostic"]

    summary = [
        "# Waterbirds finite-candidate CIC pilot (skipped)",
        "",
        "This is an optional supporting pilot, not the main result.",
        "",
        f"**{WILDS_NO_MASK_SUMMARY if (wilds_parsed and not masks_available) else reason}**",
        "",
        f"- Source: `{avail.get('source', '')}`",
        f"- Dataset available: `{dataset_available}`"
        + (f" ({avail.get('n_rows', 0)} examples parsed)" if dataset_available else ""),
        f"- Oracle-repairable masks/bboxes available: `{masks_available}`",
        f"- Oracle repair available: `{oracle_available}`",
        f"- Mask/seg/bbox-named files found in dataset tree: `{avail.get('mask_files_in_dir', 0)}`",
        f"- CLIP backend: `{status.backend}` (pretrained loaded: `{status.pretrained}`)",
        f"- Waterbirds headline eligible: `False`",
        "",
    ]
    if "wilds_converted_metadata" in extra_outputs:
        summary.append(f"- Converted WILDS metadata: `{extra_outputs['wilds_converted_metadata']}`")
    if "wilds_metadata_diagnostic" in extra_outputs:
        summary.append(
            f"- Metadata-only diagnostic (NOT CIC repair): `{extra_outputs['wilds_metadata_diagnostic']}`"
        )
    if extra_outputs:
        summary.append("")
    summary += [
        "The pilot intentionally skips oracle and failure-conditioned repair rather",
        "than fabricating a result. WILDS Waterbirds ships no bird/background masks or",
        "bounding boxes, so oracle background neutralization is not possible. Provide a",
        "dataset with segmentation masks or bird bounding boxes, then re-run. The main",
        "OpenCLIP text-overlay headline result is unaffected.",
        "",
        "This experiment only ever searches a finite, explicit candidate-intervention",
        "set. It does not perform open-world discovery and does not claim general robustness.",
    ]
    examples_md = "# Waterbirds Pilot Examples\n\nSkipped: " + reason + "\n"
    caption_md = "# Caption\n\nWaterbirds finite-candidate CIC pilot was skipped (" + reason + ").\n"
    outputs = _write_outputs(out_dir, cfg, metrics, pd.DataFrame(), pd.DataFrame(), key_numbers, summary, examples_md, caption_md)
    outputs.update(extra_outputs)
    return outputs


# ---------------------------------------------------------------------------
# WILDS conversion + metadata-only diagnostic (PART 2 / PART 4)
# ---------------------------------------------------------------------------
def _prompts(cfg: dict[str, Any]) -> list[str]:
    prompt_cfg = cfg.get("prompts", {})
    return [
        str(prompt_cfg.get("landbird", "a photo of a landbird")),
        str(prompt_cfg.get("waterbird", "a photo of a waterbird")),
    ]


def _coerce_int_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return pd.Series([pd.NA] * len(df), dtype="Int64")


def _name_of(value: Any, mapping: dict[int, str]) -> str:
    return mapping.get(int(value), "unknown") if pd.notna(value) else "unknown"


def _write_wilds_converted_metadata(
    out_dir: Path, data_cfg: dict[str, Any], avail: dict[str, Any]
) -> str | None:
    """PART 2: convert each WILDS example into explicit pilot metadata rows."""
    try:
        df = pd.read_csv(avail["metadata_csv"])
    except Exception:  # pragma: no cover - defensive
        return None
    image_col = str(data_cfg.get("image_column", "img_filename"))
    label_col = str(data_cfg.get("label_column", "y"))
    place_col = str(data_cfg.get("place_column", "place"))
    split_col = str(data_cfg.get("split_column", "split"))
    root = Path(str(avail.get("root", "")))

    y = _coerce_int_series(df, label_col)
    bg = _coerce_int_series(df, place_col)
    sp = _coerce_int_series(df, split_col)
    fnames = df[image_col].astype(str) if image_col in df.columns else pd.Series([""] * len(df))

    conv = pd.DataFrame(
        {
            "example_index": list(range(len(df))),
            "img_filename": fnames.values,
            "image_path": [str(root / f) if f else "" for f in fnames],
            "y": y.values,
            "label_name": [_name_of(v, LABEL_NAMES) for v in y],
            "background": bg.values,
            "background_name": [_name_of(v, BACKGROUND_NAMES) for v in bg],
            "split": sp.values,
            "split_name": [_name_of(v, SPLIT_NAMES) for v in sp],
        }
    )
    # Carry through optional source-domain / background-file metadata if present.
    for extra in ("from_source_domain", "source_domain", "place_filename"):
        if extra in df.columns:
            conv[extra] = df[extra].values

    path = out_dir / "wilds_converted_metadata.csv"
    conv.to_csv(path, index=False)
    return str(path)


def _load_images_no_mask(
    data_cfg: dict[str, Any], avail: dict[str, Any], size: int, seed: int
) -> list[dict[str, Any]]:
    """Load (image, label, background) tuples for the diagnostic -- no masks required."""
    root = Path(str(avail.get("root", "")))
    df = pd.read_csv(avail["metadata_csv"])
    image_col = str(data_cfg.get("image_column", "img_filename"))
    label_col = str(data_cfg.get("label_column", "y"))
    place_col = str(data_cfg.get("place_column", "place"))
    split_col = str(data_cfg.get("split_column", "split"))
    image_dir = _resolve(root, data_cfg.get("image_dir")) or root

    eval_split = data_cfg.get("eval_split", "test")
    split_map = {"train": 0, "val": 1, "validation": 1, "test": 2}
    if split_col in df.columns and eval_split is not None:
        want = split_map.get(str(eval_split).lower())
        if want is not None and (df[split_col] == want).any():
            df = df[df[split_col] == want]
    df = df.reset_index(drop=True)

    max_n = int(data_cfg.get("max_diagnostic_examples", data_cfg.get("max_natural_examples", 200)))
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=seed).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        img_path = _resolve(root, row[image_col], image_dir)
        if img_path is None or not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        except Exception:  # pragma: no cover - defensive
            continue
        label = int(row[label_col]) if label_col in df.columns and not pd.isna(row[label_col]) else -1
        if label not in (0, 1):
            continue
        place = int(row[place_col]) if place_col in df.columns and not pd.isna(row[place_col]) else -1
        rows.append({"image": img, "label": label, "place": place})
    return rows


def _metadata_diagnostic(
    out_dir: Path,
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    avail: dict[str, Any],
    status: ClipStatus,
    device: str,
) -> str | None:
    """PART 4: original CLIP accuracy / confidence by background group.

    This is a *diagnostic*, NOT CIC repair: no interventions, no oracle, no
    candidate ranking. It is never headline-eligible and is clearly marked so.
    """
    size = int(data_cfg.get("image_size", 224))
    seed = int(cfg.get("seed", 0))
    rows = _load_images_no_mask(data_cfg, avail, size, seed)
    if not rows:
        return None
    model = ClipZeroShotClassifier(status, CLASS_NAMES, prompts=_prompts(cfg), device=device)
    probs = _predict_pil(model, [r["image"] for r in rows])
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    per = pd.DataFrame(
        {
            "label": [r["label"] for r in rows],
            "background": [r["place"] for r in rows],
            "pred": preds.astype(int),
            "confidence": conf.astype(float),
            "correct": [int(int(p) == int(r["label"])) for p, r in zip(preds, rows)],
        }
    )

    out_rows: list[dict[str, Any]] = []

    def _group(name: str, sub: pd.DataFrame) -> None:
        out_rows.append(
            {
                "group": name,
                "n": int(len(sub)),
                "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan"),
                "mean_confidence": float(sub["confidence"].mean()) if len(sub) else float("nan"),
            }
        )

    _group("overall", per)
    for bg_code, bg_name in BACKGROUND_NAMES.items():
        _group(f"background={bg_name}", per[per["background"] == bg_code])
    for y_code, y_name in LABEL_NAMES.items():
        for bg_code, bg_name in BACKGROUND_NAMES.items():
            _group(f"{y_name}_on_{bg_name}", per[(per["label"] == y_code) & (per["background"] == bg_code)])

    diag = pd.DataFrame(out_rows)
    diag["diagnostic_only_not_cic_repair"] = True
    diag["headline_eligible"] = False
    diag["backend"] = status.backend
    diag["model_name"] = status.model_name
    path = out_dir / "wilds_metadata_diagnostic.csv"
    diag.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "waterbirds_cic_pilot")
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)

    # PART 1: dataset availability gate.
    avail = check_dataset(data_cfg)

    # PART 2: whenever a WILDS dataset is parsed, write the converted metadata
    # artifact (independent of whether masks/oracle repair are available).
    extra_outputs: dict[str, str] = {}
    if avail.get("dataset_available") and str(avail.get("source", "")) == "wilds":
        conv = _write_wilds_converted_metadata(out_dir, data_cfg, avail)
        if conv:
            extra_outputs["wilds_converted_metadata"] = conv

    if not avail["dataset_available"] or not avail["masks_available"]:
        # PART 4: dataset present but no masks -> optionally run a metadata-only
        # diagnostic (NOT CIC repair). Requires a real pretrained CLIP backend.
        if (
            avail.get("dataset_available")
            and bool(data_cfg.get("run_metadata_diagnostic", True))
            and str(model_cfg.get("backend", "")).lower() != "fake"
            and str(model_cfg.get("preferred_backend", "")).lower() != "fake"
        ):
            diag_status = check_clip_available(
                device=device,
                allow_download=_downloads_allowed(model_cfg),
                preferred_backend=str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip"))),
                model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
                pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
                transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
            )
            if diag_status.available and diag_status.backend in {"open_clip", "transformers"} and diag_status.pretrained:
                try:
                    diag = _metadata_diagnostic(out_dir, cfg, data_cfg, avail, diag_status, device)
                    if diag:
                        extra_outputs["wilds_metadata_diagnostic"] = diag
                except Exception:  # pragma: no cover - diagnostic is best-effort
                    pass
        status = ClipStatus(False, "not_checked", "", pretrained=False, device=device, error_message="dataset/mask gate not satisfied")
        return _write_skipped(out_dir, cfg, avail, status, avail["reason"], extra_outputs)

    # Model gate: refuse fake backends; require real pretrained CLIP.
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_waterbirds", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for Waterbirds pilot evidence")
        return _write_skipped(out_dir, cfg, avail, status, "fake CLIP backend is not allowed; pilot skipped", extra_outputs)
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_skipped(out_dir, cfg, avail, status, status.error_message or "pretrained CLIP did not load", extra_outputs)

    # prompts.
    prompts = _prompts(cfg)
    size = int(data_cfg.get("image_size", 224))
    examples = _load_examples(data_cfg, avail, size, seed)
    if len(examples) < int(data_cfg.get("min_split_examples", 4)):
        return _write_skipped(out_dir, cfg, avail, status, f"too few usable examples with masks ({len(examples)})", extra_outputs)

    model = ClipZeroShotClassifier(status, CLASS_NAMES, prompts=prompts, device=device)

    # PART 4: evaluation (Mode A natural + Mode B failure-conditioned).
    certs_nat = _evaluate(examples, model, prompts, cfg, seed)
    certs_fc = _failure_conditioned(certs_nat)

    eligible, reasons = _headline_eligibility(certs_nat, certs_fc, status, avail, cfg)
    metrics = _metric_rows(certs_nat, certs_fc, status, eligible, reasons, avail)

    def acc(method: str, df: pd.DataFrame) -> float | None:
        a, n = _accuracy(df, method)
        return None if (not np.isfinite(a) or n == 0) else float(a)

    key_numbers = {
        "dataset_available": True,
        "masks_available": True,
        "oracle_repair_available": bool(avail.get("oracle_repair_available", True)),
        "cic_repair_ran": True,
        "source": avail.get("source", "local_directory"),
        "prompts": prompts,
        "candidate_interventions": CIC_CANDIDATE_METHODS,
        "n_natural": int(len(certs_nat)),
        "n_failure_conditioned": int(len(certs_fc)),
        "original_accuracy": acc("no_intervention", certs_nat),
        "oracle_repair_accuracy": acc("oracle_background_grayfill", certs_nat),
        "oracle_blur_accuracy": acc("oracle_background_blur", certs_nat),
        "cic_top1_accuracy": acc("cic_top1", certs_nat),
        "cic_topk_accuracy": acc("cic_topk_consensus", certs_nat),
        "matched_random_accuracy": acc("random_background_patch_blur", certs_nat),
        "bird_region_blur_accuracy": acc("bird_region_blur", certs_nat),
        "bird_preservation_accuracy": acc("crop_around_bird", certs_nat),
        "failure_conditioned_original_accuracy": acc("no_intervention", certs_fc),
        "failure_conditioned_oracle_accuracy": acc("oracle_background_grayfill", certs_fc),
        "failure_conditioned_cic_top1_accuracy": acc("cic_top1", certs_fc),
        "failure_conditioned_matched_random_accuracy": acc("random_background_patch_blur", certs_fc),
        "nonoracle_scorer_excluded_label_group_correctness": True,
        "masks_used_for": "oracle repair, evaluation, and finite-candidate generation (never for non-oracle scoring)",
        "finite_candidate_not_open_world": True,
        "waterbirds_headline_eligible": bool(eligible),
        "headline_eligibility_reasons": "eligible" if eligible else reasons,
    }

    orig_acc = key_numbers["original_accuracy"]
    summary = [
        "# Waterbirds finite-candidate CIC pilot",
        "",
        "Optional supporting evidence for CIC on a real spurious-background benchmark.",
        "The causal label is bird type (landbird/waterbird); the shortcut is background",
        "habitat. This is **not** a replacement for the main OpenCLIP text-overlay result.",
        "",
        f"Evidence status: pretrained CLIP finite-candidate background pilot. Backend: `{status.backend}`. Model: `{status.model_name}`. Pretrained loaded: `{status.pretrained}`.",
        f"Waterbirds headline eligible: `{eligible}`.",
        ("Eligible." if eligible else f"Not eligible: {'; '.join(reasons)}."),
        "",
        "## Prompts",
        "",
        f"- landbird: \"{prompts[0]}\"",
        f"- waterbird: \"{prompts[1]}\"",
        "",
        "## Results (Mode A — natural held-out)",
        "",
        f"- n natural examples: {key_numbers['n_natural']}",
        f"- Original accuracy: {orig_acc if orig_acc is None else round(orig_acc, 3)}",
        f"- Oracle background neutralization accuracy: {key_numbers['oracle_repair_accuracy']}",
        f"- CIC top-1 repair accuracy: {key_numbers['cic_top1_accuracy']}",
        f"- CIC top-k consensus repair accuracy: {key_numbers['cic_topk_accuracy']}",
        f"- Matched random background repair accuracy: {key_numbers['matched_random_accuracy']}",
        f"- Bird-preservation proxy (crop around bird): {key_numbers['bird_preservation_accuracy']}",
        "",
        "## Results (Mode B — failure-conditioned)",
        "",
        f"- n verified failures: {key_numbers['n_failure_conditioned']}",
        "- Original accuracy is 0 by construction on the failure-conditioned subset.",
        f"- Oracle accuracy: {key_numbers['failure_conditioned_oracle_accuracy']}",
        f"- CIC top-1 accuracy: {key_numbers['failure_conditioned_cic_top1_accuracy']}",
        f"- Matched random accuracy: {key_numbers['failure_conditioned_matched_random_accuracy']}",
        "",
        "## Scope and integrity",
        "",
        "- Finite, explicit candidate-intervention set; **not** open-world discovery.",
        "- Non-oracle CIC scoring used only model predicted distributions: no true label,",
        "  group label, or test correctness.",
        "- Masks were used for oracle repair, evaluation, and finite-candidate generation;",
        "  they were never used in non-oracle scoring.",
        "- This pilot does not claim general robustness.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
    ]

    examples_md = "# Waterbirds Pilot Examples (first rows)\n\n" + _markdown_table(certs_nat.head(12))
    caption_md = (
        "# Caption\n\n"
        "Waterbirds finite-candidate CIC pilot: natural held-out accuracy of original CLIP, "
        "oracle background neutralization (upper bound), CIC-selected intervention, and matched "
        "random background controls. Finite-candidate intervention; not open-world discovery.\n"
    )
    outputs = _write_outputs(out_dir, cfg, metrics, certs_nat, certs_fc, key_numbers, summary, examples_md, caption_md)
    outputs.update(extra_outputs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/waterbirds_cic_pilot.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
