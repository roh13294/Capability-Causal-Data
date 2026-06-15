"""Regenerated Waterbirds-style spurious-background CIC pilot.

This is OPTIONAL supporting evidence for CIC, **not** a replacement for the main
OpenCLIP text-overlay headline result. It regenerates a Waterbirds-like
benchmark by compositing CUB-200-2011 birds (with their pixel-perfect
segmentation masks) onto Places land/water backgrounds. Because the bird mask is
known exactly, we obtain an *oracle background intervention* (neutralize the
background, keep the bird) that the previous local Waterbirds pilot needed but
could not always find.

Hard scope guarantees enforced here:
* Skips cleanly (never errors, never fabricates) if CUB images, CUB segmentation
  masks, or Places backgrounds are unavailable.
* Never silently downloads large datasets. ``data.allow_download`` defaults to
  ``False`` and only prints documented instructions when enabled.
* Never claims real Waterbirds evaluation unless actual CUB + Places assets were
  used; never claims open-world discovery; never claims exact localization;
  never claims general robustness.
* The bird/background landbird/waterbird mapping is marked **heuristic** unless an
  official mapping is found locally.
* Non-oracle CIC scoring uses ONLY model predictions -- no true label, group/
  background label, correctness, or mask leakage. Masks are used only for
  dataset generation, oracle background neutralization, evaluation/metadata, and
  *post hoc* validation of whether a candidate hit foreground/background.
* Headline eligibility is gated (PART 9). A failed gate writes a pilot/negative
  record; it never becomes a main positive result.

It does not change any existing OpenCLIP text-overlay result.
"""

from __future__ import annotations

import argparse
import hashlib
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
from PIL import Image

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

# Reuse the finite-candidate intervention bank and the non-oracle CIC scorer from
# the original Waterbirds pilot so the two pilots share identical, audited
# machinery. ``cic_rank`` is re-exported here so the leakage-signature test can
# target this module directly.
from causal_reliability.experiments.run_waterbirds_cic_pilot import (
    CIC_CANDIDATE_METHODS,
    REPORT_METHODS,
    _accuracy,
    _predict_pil,
    _wald_halfwidth,
    build_interventions,
    cic_rank,
)

__all__ = [
    "CIC_CANDIDATE_METHODS",
    "REPORT_METHODS",
    "cic_rank",
    "check_assets",
    "classify_background_folder",
    "map_cub_class_to_bird_label",
    "generate_dataset",
    "run",
]

CLASS_NAMES = ["landbird", "waterbird"]

SKIP_NO_CUB = "Regenerated Waterbirds pilot skipped: CUB-200-2011 image assets not found."
SKIP_NO_MASKS = "Regenerated Waterbirds pilot skipped: CUB segmentation masks (oracle-repairable bird/background masks) not found."
SKIP_NO_PLACES = "Regenerated Waterbirds pilot skipped: not enough Places land/water background images found."

# Heuristic water-bird keywords (PART 4). Used only when no official Waterbirds
# class mapping is found locally. Matched case-insensitively against CUB class
# names (which use underscores, e.g. ``009.Brewer_Blackbird``).
WATER_BIRD_KEYWORDS = (
    "gull",
    "tern",
    "auklet",
    "pelican",
    "cormorant",
    "frigatebird",
    "loon",
    "grebe",
    "duck",
    "goose",
    "swan",
    "albatross",
    "fulmar",
    "puffin",
    "kittiwake",
    "guillemot",
    "merganser",
    "pelagic",
    "gadwall",
    "mallard",
)

WATER_PLACE_KEYWORDS = (
    "lake",
    "ocean",
    "river",
    "beach",
    "coast",
    "waterfall",
    "pond",
    "harbor",
    "harbour",
    "swamp",
    "marsh",
    "sea",
    "wetland",
    "lagoon",
    "creek",
)
LAND_PLACE_KEYWORDS = (
    "forest",
    "field",
    "mountain",
    "desert",
    "grassland",
    "street",
    "garden",
    "park",
    "farm",
    "meadow",
    "savanna",
    "canyon",
    "orchard",
    "plain",
    "woodland",
)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# class / background label mapping (PART 4)
# ---------------------------------------------------------------------------
def map_cub_class_to_bird_label(class_name: str, official_map: dict[str, int] | None = None) -> int:
    """Return 0 (landbird) or 1 (waterbird) for a CUB class name.

    Uses an official mapping if one was supplied; otherwise a transparent
    keyword heuristic. The caller records whether the official map was used.
    """
    if official_map is not None and class_name in official_map:
        return int(official_map[class_name])
    name = class_name.lower()
    return 1 if any(k in name for k in WATER_BIRD_KEYWORDS) else 0


def classify_background_folder(folder_name: str) -> str | None:
    """Classify a Places folder/scene name into 'water', 'land', or None."""
    name = folder_name.lower()
    if any(k in name for k in WATER_PLACE_KEYWORDS):
        return "water"
    if any(k in name for k in LAND_PLACE_KEYWORDS):
        return "land"
    return None


def _load_official_map(cub_root: Path, data_cfg: dict[str, Any]) -> dict[str, int] | None:
    """Best-effort load of an official landbird/waterbird mapping if present.

    Looks for an optional ``waterbird_classes.txt`` (one CUB class name per line,
    optionally ``<name> <0|1>``). Returns None if absent -> heuristic mapping.
    """
    candidates: list[Path] = []
    explicit = data_cfg.get("official_class_map")
    if explicit:
        candidates.append(Path(str(explicit)))
    candidates.append(cub_root / "waterbird_classes.txt")
    for path in candidates:
        if not path.exists():
            continue
        mapping: dict[str, int] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[-1] in {"0", "1"}:
                    mapping[" ".join(parts[:-1])] = int(parts[-1])
                else:
                    mapping[line] = 1  # listed = waterbird
            if mapping:
                return mapping
        except Exception:  # pragma: no cover - defensive
            continue
    return None


# ---------------------------------------------------------------------------
# asset availability gate (PARTS 2 + 4)
# ---------------------------------------------------------------------------
def _gather_backgrounds(places_root: Path, min_per_type: int) -> dict[str, list[str]]:
    """Walk the Places root, classifying images by their containing folder name."""
    by_type: dict[str, list[str]] = {"water": [], "land": []}
    if not places_root.exists():
        return by_type
    for dirpath, _dirnames, filenames in os.walk(places_root):
        folder = Path(dirpath).name
        bg_type = classify_background_folder(folder)
        if bg_type is None:
            continue
        for fname in filenames:
            if fname.lower().endswith(_IMAGE_EXTS):
                by_type[bg_type].append(str(Path(dirpath) / fname))
        # Early exit once we have plenty of both, to bound the walk.
        if len(by_type["water"]) > 50 * max(1, min_per_type) and len(by_type["land"]) > 50 * max(1, min_per_type):
            break
    return by_type


def check_assets(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Structured availability report for CUB + segmentations + Places.

    Never raises. ``missing`` lists every required path that was absent.
    """
    cub_root = Path(str(data_cfg.get("cub_root", "data/cub/CUB_200_2011")))
    seg_root = Path(str(data_cfg.get("cub_segmentations_root", "data/cub/segmentations")))
    places_root = Path(str(data_cfg.get("places_root", "data/places")))
    min_bg = int(data_cfg.get("min_backgrounds_per_type", 5))

    required_cub = {
        "images.txt": cub_root / "images.txt",
        "image_class_labels.txt": cub_root / "image_class_labels.txt",
        "classes.txt": cub_root / "classes.txt",
        "bounding_boxes.txt": cub_root / "bounding_boxes.txt",
        "images/": cub_root / "images",
    }
    missing: list[str] = [name for name, path in required_cub.items() if not path.exists()]

    report: dict[str, Any] = {
        "cub_available": False,
        "cub_segmentations_available": False,
        "places_available": False,
        "cub_root": str(cub_root),
        "segmentations_root": str(seg_root),
        "places_root": str(places_root),
        "missing": missing,
        "n_water_backgrounds": 0,
        "n_land_backgrounds": 0,
        "min_backgrounds_per_type": min_bg,
        "reason": "",
    }

    if missing:
        report["reason"] = f"{SKIP_NO_CUB} Missing: {', '.join(missing)}"
        return report
    report["cub_available"] = True

    if not seg_root.exists() or not any(seg_root.rglob("*.png")):
        report["missing"] = list(missing) + [f"segmentations under {seg_root}"]
        report["reason"] = SKIP_NO_MASKS
        return report
    report["cub_segmentations_available"] = True

    backgrounds = _gather_backgrounds(places_root, min_bg)
    report["n_water_backgrounds"] = len(backgrounds["water"])
    report["n_land_backgrounds"] = len(backgrounds["land"])
    if report["n_water_backgrounds"] < min_bg or report["n_land_backgrounds"] < min_bg:
        report["missing"] = list(missing) + [
            f"Places water backgrounds (have {report['n_water_backgrounds']}, need {min_bg})",
            f"Places land backgrounds (have {report['n_land_backgrounds']}, need {min_bg})",
        ]
        report["reason"] = SKIP_NO_PLACES
        return report
    report["places_available"] = True
    report["reason"] = "available"
    report["_backgrounds"] = backgrounds  # private handle for generation
    return report


# ---------------------------------------------------------------------------
# CUB index parsing
# ---------------------------------------------------------------------------
def _read_id_table(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            try:
                out[int(parts[0])] = parts[1]
            except ValueError:
                continue
    return out


def _read_bboxes(path: Path) -> dict[int, tuple[float, float, float, float]]:
    out: dict[int, tuple[float, float, float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                i = int(parts[0])
                x, y, w, h = (float(v) for v in parts[1:5])
                out[i] = (x, y, w, h)
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------------------
# compositing (PART 4)
# ---------------------------------------------------------------------------
def _composite(bird_rgb: np.ndarray, bird_mask: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """out = bird where mask else background. All inputs float in [0,1], HxWx3."""
    m = bird_mask[..., None].astype(np.float32)
    return bird_rgb * m + bg_rgb * (1.0 - m)


def _prepare_bird(img: Image.Image, mask: Image.Image, bbox: tuple[float, float, float, float], size: int, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Crop around the bbox with margin, then resize bird image + mask to a square."""
    W, H = img.size
    x, y, w, h = bbox
    mx, my = margin * w, margin * h
    x0 = int(max(0, math.floor(x - mx)))
    y0 = int(max(0, math.floor(y - my)))
    x1 = int(min(W, math.ceil(x + w + mx)))
    y1 = int(min(H, math.ceil(y + h + my)))
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, W, H
    bird = img.crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.BICUBIC)
    mcrop = mask.crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.NEAREST)
    bird_rgb = np.asarray(bird.convert("RGB")).astype(np.float32) / 255.0
    bird_mask = np.asarray(mcrop.convert("L")) > 127
    return bird_rgb, bird_mask


def _resize_bg(path: str, size: int) -> np.ndarray | None:
    try:
        bg = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    except Exception:
        return None
    return np.asarray(bg).astype(np.float32) / 255.0


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def generate_dataset(
    avail: dict[str, Any],
    data_cfg: dict[str, Any],
    out_root: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Build the regenerated Waterbirds-style composites + metadata.

    Returns ``(examples, metadata_df, gen_info)``. Each example carries the
    aligned/misleading/neutral composites for one bird at a fixed paste location
    (so a single bird mask applies to all three), plus labels.
    """
    cub_root = Path(avail["cub_root"])
    seg_root = Path(avail["segmentations_root"])
    backgrounds: dict[str, list[str]] = avail.get("_backgrounds") or _gather_backgrounds(Path(avail["places_root"]), 1)
    size = int(data_cfg.get("image_size", 224))
    margin = float(data_cfg.get("bird_crop_margin", 0.15))
    max_examples = int(data_cfg.get("max_examples", 400))
    gray_fill = float(data_cfg.get("neutral_fill_gray", 0.5))

    images = _read_id_table(cub_root / "images.txt")
    labels = {k: int(v) for k, v in _read_id_table(cub_root / "image_class_labels.txt").items()}
    classes = _read_id_table(cub_root / "classes.txt")
    bboxes = _read_bboxes(cub_root / "bounding_boxes.txt")

    official_map_raw = _load_official_map(cub_root, data_cfg)
    official_map: dict[str, int] | None = None
    if official_map_raw is not None:
        # Re-key by class name; class file values are like "001.Black_footed_Albatross".
        official_map = official_map_raw
    used_official = official_map is not None

    rng = np.random.default_rng(seed)
    image_ids = sorted(images.keys())
    rng.shuffle(image_ids)

    img_dir = cub_root / "images"
    examples: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    images_dir = ensure_dir(out_root / "images")
    masks_dir = ensure_dir(out_root / "masks")

    bg_idx = {"water": 0, "land": 0}
    rng.shuffle(backgrounds["water"])
    rng.shuffle(backgrounds["land"])

    def _next_bg(bg_type: str) -> np.ndarray | None:
        pool = backgrounds.get(bg_type) or []
        for _ in range(min(len(pool), 8)):
            if not pool:
                return None
            path = pool[bg_idx[bg_type] % len(pool)]
            bg_idx[bg_type] += 1
            arr = _resize_bg(path, size)
            if arr is not None:
                return arr
        return None

    for image_id in image_ids:
        if len(examples) >= max_examples:
            break
        rel = images.get(image_id)
        cls_id = labels.get(image_id)
        bbox = bboxes.get(image_id)
        if rel is None or cls_id is None or bbox is None:
            continue
        class_name = classes.get(cls_id, str(cls_id))
        bird_label = map_cub_class_to_bird_label(class_name, official_map)
        img_path = img_dir / rel
        mask_path = seg_root / rel
        if not mask_path.exists():
            mask_path = seg_root / (str(rel).rsplit(".", 1)[0] + ".png")
        if not img_path.exists() or not mask_path.exists():
            continue
        try:
            bird_img = Image.open(img_path).convert("RGB")
            bird_seg = Image.open(mask_path).convert("L")
        except Exception:
            continue
        bird_rgb, bird_mask = _prepare_bird(bird_img, bird_seg, bbox, size, margin)
        if not bird_mask.any() or bird_mask.all():
            continue  # need a usable bird/background separation

        # waterbird (1) -> water habitat aligned; landbird (0) -> land aligned.
        aligned_type = "water" if bird_label == 1 else "land"
        misleading_type = "land" if bird_label == 1 else "water"
        bg_aligned = _next_bg(aligned_type)
        bg_misleading = _next_bg(misleading_type)
        if bg_aligned is None or bg_misleading is None:
            continue

        aligned_arr = _composite(bird_rgb, bird_mask, bg_aligned)
        misleading_arr = _composite(bird_rgb, bird_mask, bg_misleading)
        neutral_arr = _composite(bird_rgb, bird_mask, np.full_like(bird_rgb, gray_fill))

        ex_id = len(examples)
        aligned_pil = _to_pil(aligned_arr)
        misleading_pil = _to_pil(misleading_arr)
        neutral_pil = _to_pil(neutral_arr)
        mask_u8 = (bird_mask.astype(np.uint8) * 255)

        a_path = images_dir / f"ex{ex_id:05d}_aligned.png"
        m_path = images_dir / f"ex{ex_id:05d}_misleading.png"
        n_path = images_dir / f"ex{ex_id:05d}_neutral.png"
        bm_path = masks_dir / f"ex{ex_id:05d}_bird_mask.png"
        bgm_path = masks_dir / f"ex{ex_id:05d}_background_mask.png"
        aligned_pil.save(a_path)
        misleading_pil.save(m_path)
        neutral_pil.save(n_path)
        Image.fromarray(mask_u8).save(bm_path)
        Image.fromarray(255 - mask_u8).save(bgm_path)

        examples.append(
            {
                "example_id": ex_id,
                "aligned_image": aligned_pil,
                "misleading_image": misleading_pil,
                "neutral_image": neutral_pil,
                "mask": bird_mask,
                "label": int(bird_label),
                "aligned_background": aligned_type,
                "misleading_background": misleading_type,
            }
        )

        img_hash = _hash_array(np.asarray(misleading_pil))
        mask_hash = _hash_array(mask_u8)
        for regime, path, bg_label, aligned_or in (
            ("aligned", a_path, aligned_type, "aligned"),
            ("misleading", m_path, misleading_type, "misleading"),
            ("neutral", n_path, "neutral", "neutral"),
        ):
            meta_rows.append(
                {
                    "example_id": ex_id,
                    "regime": regime,
                    "image_path": str(path),
                    "bird_mask_path": str(bm_path),
                    "background_mask_path": str(bgm_path),
                    "bird_label": CLASS_NAMES[bird_label],
                    "background_label": bg_label,
                    "aligned_or_misleading": aligned_or,
                    "source_cub_image": str(img_path),
                    "source_cub_class": class_name,
                    "source_places_image": "neutral_gray" if regime == "neutral" else "",
                    "source_places_class": bg_label,
                    "split": "test",
                    "seed": seed,
                    "image_hash": _hash_array(np.asarray(aligned_pil)) if regime == "aligned" else (img_hash if regime == "misleading" else _hash_array(np.asarray(neutral_pil))),
                    "mask_hash": mask_hash,
                }
            )

    metadata_df = pd.DataFrame(meta_rows)
    gen_info = {
        "n_examples": len(examples),
        "official_class_map_used": bool(used_official),
        "class_map_kind": "official" if used_official else "heuristic_keyword",
        "bird_crop_margin": margin,
        "image_size": size,
    }
    return examples, metadata_df, gen_info


# ---------------------------------------------------------------------------
# regime-aware evaluation (PARTS 5-8)
# ---------------------------------------------------------------------------
def _evaluate(
    examples: list[dict[str, Any]],
    model: ClipZeroShotClassifier,
    cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Aligned image is scored directly; CIC/oracle repairs run on the MISLEADING
    image (where the background shortcut hurts). No labels reach the scorer."""
    rng = np.random.default_rng(seed)
    cfg_iv = cfg.get("intervention", {})
    top_k = int(cfg_iv.get("top_k", 3))
    rows: list[dict[str, Any]] = []
    for ex in examples:
        label = ex["label"]
        # Aligned + neutral predictions (single forward each).
        extra = _predict_pil(model, [ex["aligned_image"], ex["neutral_image"]])
        aligned_pred = int(np.argmax(extra[0]))
        neutral_pred = int(np.argmax(extra[1]))

        # Build finite-candidate interventions on the misleading image.
        iv_ex = {"image": ex["misleading_image"], "mask": ex["mask"], "label": label, "example_id": ex["example_id"]}
        interventions = build_interventions(iv_ex, cfg_iv, rng)
        names = list(interventions.keys())
        probs_arr = _predict_pil(model, [interventions[n] for n in names])
        probs = {n: probs_arr[i] for i, n in enumerate(names)}
        original_probs = probs["no_intervention"]

        candidate_probs = {n: probs[n] for n in CIC_CANDIDATE_METHODS if n in probs}
        ranked = cic_rank(candidate_probs, original_probs)
        cic_top1_name = ranked[0][0] if ranked else "no_intervention"
        cic_topk_names = [n for n, _ in ranked[:top_k]]
        topk_preds = [int(np.argmax(probs[n])) for n in cic_topk_names]
        if topk_preds:
            vals, counts = np.unique(topk_preds, return_counts=True)
            cic_topk_pred = int(vals[int(counts.argmax())])
        else:
            cic_topk_pred = int(np.argmax(original_probs))

        row: dict[str, Any] = {
            "example_id": ex["example_id"],
            "label": label,
            "aligned_background": ex["aligned_background"],
            "misleading_background": ex["misleading_background"],
            "aligned__pred": aligned_pred,
            "aligned__correct": bool(aligned_pred == label),
            "neutral__pred": neutral_pred,
            "neutral__correct": bool(neutral_pred == label),
            "misleading__conf": float(np.max(original_probs)),
            "cic_top1_selected": cic_top1_name,
            "cic_topk_selected": "|".join(cic_topk_names),
        }
        for name in names:
            pred = int(np.argmax(probs[name]))
            row[f"{name}__pred"] = pred
            row[f"{name}__conf"] = float(np.max(probs[name]))
            row[f"{name}__correct"] = bool(pred == label)
        row["cic_top1__pred"] = int(np.argmax(probs[cic_top1_name]))
        row["cic_top1__correct"] = bool(int(np.argmax(probs[cic_top1_name])) == label)
        row["cic_topk_consensus__pred"] = cic_topk_pred
        row["cic_topk_consensus__correct"] = bool(cic_topk_pred == label)
        rows.append(row)
    return pd.DataFrame(rows)


def _failure_conditioned(certs: pd.DataFrame, conf_threshold: float) -> pd.DataFrame:
    """Mode B: aligned correct, misleading wrong, confident, oracle restores."""
    if len(certs) == 0:
        return certs
    cond = (
        certs["aligned__correct"].astype(bool)
        & (~certs["no_intervention__correct"].astype(bool))
        & (certs["misleading__conf"].astype(float) >= float(conf_threshold))
        & certs["oracle_background_grayfill__correct"].astype(bool)
    )
    return certs[cond].reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics + eligibility (PART 9)
# ---------------------------------------------------------------------------
def _metric_rows(
    certs_nat: pd.DataFrame,
    certs_fc: pd.DataFrame,
    status: ClipStatus,
    headline: bool,
    reasons: list[str],
    avail: dict[str, Any],
    gen_info: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evidence = "pretrained CLIP regenerated-Waterbirds background pilot" if status.pretrained else "unavailable"
    methods = ["aligned", "neutral"] + REPORT_METHODS
    for method in methods:
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
                "cub_available": bool(avail.get("cub_available", False)),
                "cub_segmentations_available": bool(avail.get("cub_segmentations_available", False)),
                "places_available": bool(avail.get("places_available", False)),
                "class_map_kind": gen_info.get("class_map_kind", "heuristic_keyword"),
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
    gen_info: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    elig = cfg.get("eligibility", {})
    if status.backend not in {"open_clip", "transformers"} or not status.pretrained:
        reasons.append("real pretrained OpenCLIP did not load")
    if status.backend == "fake":
        reasons.append("fake backend cannot be headline eligible")
    if not avail.get("cub_available", False):
        reasons.append("actual CUB images unavailable")
    if not avail.get("cub_segmentations_available", False):
        reasons.append("actual CUB segmentation masks unavailable")
    if not avail.get("places_available", False):
        reasons.append("actual Places backgrounds unavailable")
    if len(certs_nat) == 0:
        reasons.append("no evaluated examples")
        return False, reasons

    aligned_acc, _ = _accuracy(certs_nat, "aligned")
    mis_acc, _ = _accuracy(certs_nat, "no_intervention")
    oracle_acc, _ = _accuracy(certs_nat, "oracle_background_grayfill")
    cic1_acc, _ = _accuracy(certs_nat, "cic_top1")
    cick_acc, _ = _accuracy(certs_nat, "cic_topk_consensus")
    rand_acc, rand_n = _accuracy(certs_nat, "random_background_patch_blur")
    n_nat = int(len(certs_nat))
    n_fc = int(len(certs_fc))

    min_nat = int(elig.get("min_natural", 100))
    min_fc = int(elig.get("min_failure_conditioned", 30))
    if not (n_nat >= min_nat or n_fc >= min_fc):
        reasons.append(f"insufficient examples (natural n={n_nat} < {min_nat} and failure-conditioned n={n_fc} < {min_fc})")

    mis_gap = float(elig.get("misleading_below_aligned_abs", 0.10))
    if not (np.isfinite(aligned_acc) and np.isfinite(mis_acc) and (aligned_acc - mis_acc) >= mis_gap):
        reasons.append(
            f"misleading accuracy not meaningfully below aligned (aligned {aligned_acc:.3f} vs misleading {mis_acc:.3f}, gap < {mis_gap})"
        )

    oracle_gain = float(elig.get("oracle_min_gain_abs", 0.15))
    oracle_fc_acc, _ = _accuracy(certs_fc, "oracle_background_grayfill")
    fc_restore = float(elig.get("oracle_fc_restore", 0.80))
    oracle_ok = (np.isfinite(oracle_acc) and np.isfinite(mis_acc) and (oracle_acc - mis_acc) >= oracle_gain) or (
        n_fc > 0 and np.isfinite(oracle_fc_acc) and oracle_fc_acc >= fc_restore
    )
    if not oracle_ok:
        reasons.append(
            f"oracle background neutralization did not improve >= {oracle_gain} (misleading {mis_acc:.3f} -> oracle {oracle_acc:.3f}) "
            f"nor restore >= {fc_restore} on failure-conditioned (oracle_fc {oracle_fc_acc})"
        )

    best_cic = max([a for a in [cic1_acc, cick_acc] if np.isfinite(a)] or [float("nan")])
    margin_req = float(elig.get("cic_beats_random_abs", 0.15))
    beats_abs = np.isfinite(best_cic) and np.isfinite(rand_acc) and (best_cic - rand_acc) >= margin_req
    ci_nonoverlap = False
    if np.isfinite(best_cic) and np.isfinite(rand_acc):
        lo_cic = best_cic - _wald_halfwidth(best_cic, n_nat)
        hi_rand = rand_acc + _wald_halfwidth(rand_acc, rand_n)
        ci_nonoverlap = lo_cic > hi_rand
    if not (beats_abs or ci_nonoverlap):
        reasons.append(
            f"CIC does not beat matched random by >= {margin_req} (best CIC {best_cic:.3f} vs random {rand_acc:.3f})"
        )

    min_pres = float(elig.get("min_aligned_preservation", 0.70))
    if not (np.isfinite(aligned_acc) and aligned_acc >= min_pres):
        reasons.append(f"clean/aligned preservation {aligned_acc:.3f} < {min_pres}")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# output writers (PART 10)
# ---------------------------------------------------------------------------
def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    plt.figure(figsize=(10.4, 4.8))
    plottable = metrics[metrics.get("natural_accuracy", pd.Series(dtype=float)).notna()] if len(metrics) else pd.DataFrame()
    if len(plottable):
        x = np.arange(len(plottable))
        colors = []
        for m in plottable["method"]:
            if "cic" in m:
                colors.append("#4c78a8")
            elif "oracle" in m or m == "aligned":
                colors.append("#54a24b")
            else:
                colors.append("#bab0ac")
        plt.bar(x, plottable["natural_accuracy"], color=colors)
        if "natural_accuracy_ci95_halfwidth" in plottable:
            plt.errorbar(x, plottable["natural_accuracy"], yerr=plottable["natural_accuracy_ci95_halfwidth"], fmt="none", ecolor="#333333", capsize=3)
        plt.xticks(x, plottable["method"], rotation=35, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("Accuracy")
        plt.title("Regenerated Waterbirds-style CIC pilot (CUB + Places)")
    else:
        plt.text(0.5, 0.5, "Regenerated Waterbirds pilot skipped (no CUB / masks / Places / model)", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _artifact_paths(out_dir: Path) -> dict[str, str]:
    return {
        "summary": str(out_dir / "regenerated_waterbirds_summary.md"),
        "metrics": str(out_dir / "regenerated_waterbirds_metrics.csv"),
        "key_numbers": str(out_dir / "regenerated_waterbirds_key_numbers.json"),
        "certificates": str(out_dir / "regenerated_waterbirds_certificates.csv"),
        "examples": str(out_dir / "regenerated_waterbirds_examples.md"),
        "caption": str(out_dir / "regenerated_waterbirds_caption.md"),
        "plot": str(out_dir / "regenerated_waterbirds_plot.png"),
        "config_used": str(out_dir / "regenerated_waterbirds_config_used.yaml"),
    }


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
    paths = _artifact_paths(out_dir)
    metrics.to_csv(paths["metrics"], index=False)
    certs_nat.to_csv(paths["certificates"], index=False)
    certs_fc.to_csv(out_dir / "regenerated_waterbirds_failure_conditioned_certificates.csv", index=False)
    Path(paths["key_numbers"]).write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")
    Path(paths["config_used"]).write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    Path(paths["summary"]).write_text("\n".join(summary_lines), encoding="utf-8")
    Path(paths["examples"]).write_text(examples_md, encoding="utf-8")
    Path(paths["caption"]).write_text(caption_md, encoding="utf-8")
    _plot(metrics, out_dir / "regenerated_waterbirds_plot.png", out_dir / "regenerated_waterbirds_plot.pdf")
    return paths


def _write_skipped(out_dir: Path, cfg: dict[str, Any], avail: dict[str, Any], status: ClipStatus, reason: str) -> dict[str, str]:
    """Write the regeneration-skip record (PART 2)."""
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
                "cub_available": bool(avail.get("cub_available", False)),
                "cub_segmentations_available": bool(avail.get("cub_segmentations_available", False)),
                "places_available": bool(avail.get("places_available", False)),
                "class_map_kind": "n/a",
                "finite_candidate_not_open_world": True,
                "natural_n": 0,
                "natural_accuracy": float("nan"),
                "failure_conditioned_n": 0,
                "failure_conditioned_accuracy": float("nan"),
            }
        ]
    )
    key_numbers = {
        "regenerated_waterbirds_available": False,
        "cub_available": bool(avail.get("cub_available", False)),
        "cub_segmentations_available": bool(avail.get("cub_segmentations_available", False)),
        "places_available": bool(avail.get("places_available", False)),
        "missing": avail.get("missing", []),
        "n_natural": 0,
        "n_failure_conditioned": 0,
        "dataset_generated": False,
        "original_accuracy": None,
        "aligned_accuracy": None,
        "misleading_accuracy": None,
        "oracle_repair_accuracy": None,
        "cic_top1_accuracy": None,
        "cic_topk_accuracy": None,
        "matched_random_accuracy": None,
        "nonoracle_scorer_excluded_label_group_masks_correctness": True,
        "masks_used_for": "n/a (skipped)",
        "waterbirds_headline_eligible": False,
        "regenerated_waterbirds_headline_eligible": False,
        "skipped": True,
        "skip_reason": reason,
        "finite_candidate_not_open_world": True,
    }
    missing_lines = [f"- `{m}`" for m in avail.get("missing", [])] or ["- (none recorded)"]
    # PART 2 also mandates the dedicated regeneration-summary artifacts.
    (out_dir / "waterbirds_regeneration_summary.md").write_text(
        "\n".join(
            [
                "# Regenerated Waterbirds-style regeneration (skipped)",
                "",
                f"**{reason}**",
                "",
                f"- CUB available: `{avail.get('cub_available', False)}`",
                f"- CUB segmentations available: `{avail.get('cub_segmentations_available', False)}`",
                f"- Places backgrounds available: `{avail.get('places_available', False)}`",
                f"- Water backgrounds found: `{avail.get('n_water_backgrounds', 0)}`",
                f"- Land backgrounds found: `{avail.get('n_land_backgrounds', 0)}`",
                "",
                "## Missing paths",
                "",
                *missing_lines,
                "",
                "See `docs/regenerated_waterbirds_pilot.md` and the README for where to place",
                "CUB images, CUB segmentations, and Places backgrounds. The main OpenCLIP",
                "text-overlay headline result is unaffected.",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "waterbirds_regeneration_key_numbers.json").write_text(
        json.dumps(
            {
                "regenerated_waterbirds_available": False,
                "waterbirds_headline_eligible": False,
                "regenerated_waterbirds_headline_eligible": False,
                "cub_available": bool(avail.get("cub_available", False)),
                "cub_segmentations_available": bool(avail.get("cub_segmentations_available", False)),
                "places_available": bool(avail.get("places_available", False)),
                "missing": avail.get("missing", []),
                "skip_reason": reason,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = [
        "# Regenerated Waterbirds-style spurious-background CIC pilot (skipped)",
        "",
        "Optional supporting pilot, not the main result.",
        "",
        f"**{reason}**",
        "",
        f"- CUB available: `{avail.get('cub_available', False)}`",
        f"- CUB segmentations available: `{avail.get('cub_segmentations_available', False)}`",
        f"- Places backgrounds available: `{avail.get('places_available', False)}`",
        f"- CLIP backend: `{status.backend}` (pretrained loaded: `{status.pretrained}`)",
        "- Regenerated Waterbirds headline eligible: `False`",
        "",
        "## Missing paths",
        "",
        *missing_lines,
        "",
        "The pilot intentionally skips rather than fabricating a result. Provide CUB-200-2011",
        "(`data/cub/CUB_200_2011`), CUB segmentation masks (`data/cub/segmentations`), and",
        "Places land/water backgrounds (`data/places`), then re-run. This is a controlled",
        "**regenerated Waterbirds-style** benchmark, not open-world shortcut discovery, and it",
        "does not change the main OpenCLIP text-overlay headline.",
    ]
    examples_md = "# Regenerated Waterbirds Examples\n\nSkipped: " + reason + "\n"
    caption_md = "# Caption\n\nRegenerated Waterbirds-style CIC pilot skipped (" + reason + ").\n"
    return _write_outputs(out_dir, cfg, metrics, pd.DataFrame(), pd.DataFrame(), key_numbers, summary, examples_md, caption_md)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "regenerated_waterbirds_cic")
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)

    # PART 3: never silently download. If allow_download is set, print the
    # documented instructions but do not scrape or fetch huge datasets here.
    if bool(data_cfg.get("allow_download", False)):
        print(
            "[regenerated_waterbirds] data.allow_download=true: automatic dataset download is "
            "intentionally NOT implemented. Place CUB-200-2011 under data/cub/CUB_200_2011, CUB "
            "segmentations under data/cub/segmentations, and Places backgrounds under data/places. "
            "See docs/regenerated_waterbirds_pilot.md."
        )

    # PART 2/4: asset availability gate.
    avail = check_assets(data_cfg)
    if not (avail["cub_available"] and avail["cub_segmentations_available"] and avail["places_available"]):
        status = ClipStatus(False, "not_checked", "", pretrained=False, device=device, error_message="CUB/Places asset gate not satisfied")
        return _write_skipped(out_dir, cfg, avail, status, avail["reason"])

    # Model gate: refuse fake backends; require real pretrained CLIP for headline.
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_regenerated_waterbirds", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for the regenerated Waterbirds pilot")
        return _write_skipped(out_dir, cfg, avail, status, "fake CLIP backend is not allowed; pilot skipped")
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_skipped(out_dir, cfg, avail, status, status.error_message or "pretrained CLIP did not load")

    # PART 4: regenerate the dataset.
    out_root = ensure_dir(Path(str(data_cfg.get("output_root", "data/regenerated_waterbirds"))))
    examples, metadata_df, gen_info = generate_dataset(avail, data_cfg, out_root, seed)
    metadata_df.to_csv(out_root / "metadata.csv", index=False)
    if len(examples) < int(data_cfg.get("min_examples", 4)):
        return _write_skipped(out_dir, cfg, avail, status, f"too few usable composites generated ({len(examples)})")

    # PART 5: prompts (one predeclared primary pair).
    prompt_cfg = cfg.get("prompts", {})
    prompts = [
        str(prompt_cfg.get("landbird", "a photo of a landbird")),
        str(prompt_cfg.get("waterbird", "a photo of a waterbird")),
    ]
    model = ClipZeroShotClassifier(status, CLASS_NAMES, prompts=prompts, device=device)

    # PARTS 5-8: evaluation (Mode A natural + Mode B failure-conditioned).
    conf_threshold = float(cfg.get("eligibility", {}).get("failure_confidence_threshold", 0.50))
    certs_nat = _evaluate(examples, model, cfg, seed)
    certs_fc = _failure_conditioned(certs_nat, conf_threshold)

    eligible, reasons = _headline_eligibility(certs_nat, certs_fc, status, avail, gen_info, cfg)
    metrics = _metric_rows(certs_nat, certs_fc, status, eligible, reasons, avail, gen_info)

    def acc(method: str, df: pd.DataFrame) -> float | None:
        a, n = _accuracy(df, method)
        return None if (not np.isfinite(a) or n == 0) else float(a)

    key_numbers = {
        "regenerated_waterbirds_available": True,
        "dataset_generated": True,
        "cub_available": True,
        "cub_segmentations_available": True,
        "places_available": True,
        "n_water_backgrounds": int(avail.get("n_water_backgrounds", 0)),
        "n_land_backgrounds": int(avail.get("n_land_backgrounds", 0)),
        "class_map_kind": gen_info.get("class_map_kind", "heuristic_keyword"),
        "official_class_map_used": bool(gen_info.get("official_class_map_used", False)),
        "prompts": prompts,
        "candidate_interventions": CIC_CANDIDATE_METHODS,
        "n_natural": int(len(certs_nat)),
        "n_failure_conditioned": int(len(certs_fc)),
        "failure_confidence_threshold": conf_threshold,
        "aligned_accuracy": acc("aligned", certs_nat),
        "neutral_accuracy": acc("neutral", certs_nat),
        "misleading_accuracy": acc("no_intervention", certs_nat),
        "original_accuracy": acc("no_intervention", certs_nat),
        "oracle_repair_accuracy": acc("oracle_background_grayfill", certs_nat),
        "oracle_blur_accuracy": acc("oracle_background_blur", certs_nat),
        "cic_top1_accuracy": acc("cic_top1", certs_nat),
        "cic_topk_accuracy": acc("cic_topk_consensus", certs_nat),
        "matched_random_accuracy": acc("random_background_patch_blur", certs_nat),
        "bird_region_blur_accuracy": acc("bird_region_blur", certs_nat),
        "failure_conditioned_original_accuracy": acc("no_intervention", certs_fc),
        "failure_conditioned_oracle_accuracy": acc("oracle_background_grayfill", certs_fc),
        "failure_conditioned_cic_top1_accuracy": acc("cic_top1", certs_fc),
        "failure_conditioned_cic_topk_accuracy": acc("cic_topk_consensus", certs_fc),
        "failure_conditioned_matched_random_accuracy": acc("random_background_patch_blur", certs_fc),
        "failure_conditioned_inclusion_rate": (float(len(certs_fc) / len(certs_nat)) if len(certs_nat) else None),
        "nonoracle_scorer_excluded_label_group_masks_correctness": True,
        "masks_used_for": "dataset generation, oracle background neutralization, evaluation/metadata, and post-hoc candidate validation (never for non-oracle scoring)",
        "finite_candidate_not_open_world": True,
        "waterbirds_headline_eligible": bool(eligible),
        "regenerated_waterbirds_headline_eligible": bool(eligible),
        "headline_eligibility_reasons": "eligible" if eligible else reasons,
        "metadata_csv": str(out_root / "metadata.csv"),
    }
    # PART 2 dedicated regeneration artifacts (success path).
    (out_dir / "waterbirds_regeneration_summary.md").write_text(
        "\n".join(
            [
                "# Regenerated Waterbirds-style regeneration",
                "",
                f"- Composites generated: `{len(examples)}`",
                f"- Class mapping: `{gen_info.get('class_map_kind')}` (official used: `{gen_info.get('official_class_map_used')}`)",
                f"- Water backgrounds available: `{avail.get('n_water_backgrounds', 0)}`",
                f"- Land backgrounds available: `{avail.get('n_land_backgrounds', 0)}`",
                f"- Metadata: `{out_root / 'metadata.csv'}`",
                f"- Regenerated Waterbirds headline eligible: `{eligible}`",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "waterbirds_regeneration_key_numbers.json").write_text(
        json.dumps(
            {
                "regenerated_waterbirds_available": True,
                "n_examples": len(examples),
                "class_map_kind": gen_info.get("class_map_kind"),
                "regenerated_waterbirds_headline_eligible": bool(eligible),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary = [
        "# Regenerated Waterbirds-style spurious-background CIC pilot",
        "",
        "Optional supporting evidence for CIC on a **controlled regenerated Waterbirds-style",
        "benchmark** built by compositing CUB-200-2011 birds onto Places land/water",
        "backgrounds. The bird segmentation mask is known exactly, giving an oracle background",
        "intervention. This is **not** a replacement for the main OpenCLIP text-overlay result,",
        "and it is **not** open-world shortcut discovery.",
        "",
        f"Evidence status: pretrained CLIP regenerated-Waterbirds pilot. Backend: `{status.backend}`. Model: `{status.model_name}`. Pretrained loaded: `{status.pretrained}`.",
        f"Class mapping: `{gen_info.get('class_map_kind')}` (official used: `{gen_info.get('official_class_map_used')}`).",
        f"Regenerated Waterbirds headline eligible: `{eligible}`.",
        ("Eligible." if eligible else f"Not eligible: {'; '.join(reasons)}."),
        "",
        "## Prompts",
        "",
        f"- landbird: \"{prompts[0]}\"",
        f"- waterbird: \"{prompts[1]}\"",
        "",
        "## Results (Mode A — natural regenerated benchmark)",
        "",
        f"- n examples: {key_numbers['n_natural']}",
        f"- Aligned accuracy: {key_numbers['aligned_accuracy']}",
        f"- Misleading accuracy: {key_numbers['misleading_accuracy']}",
        f"- Neutral-background accuracy: {key_numbers['neutral_accuracy']}",
        f"- Oracle background-neutralized accuracy: {key_numbers['oracle_repair_accuracy']}",
        f"- CIC top-1 repair accuracy: {key_numbers['cic_top1_accuracy']}",
        f"- CIC top-k consensus repair accuracy: {key_numbers['cic_topk_accuracy']}",
        f"- Matched random background repair accuracy: {key_numbers['matched_random_accuracy']}",
        "",
        "## Results (Mode B — failure-conditioned)",
        "",
        f"- n verified failures: {key_numbers['n_failure_conditioned']}",
        f"- Inclusion rate: {key_numbers['failure_conditioned_inclusion_rate']}",
        "- Original accuracy is 0 by construction on the failure-conditioned subset.",
        f"- Oracle repair accuracy: {key_numbers['failure_conditioned_oracle_accuracy']}",
        f"- CIC top-1 repair accuracy: {key_numbers['failure_conditioned_cic_top1_accuracy']}",
        f"- Matched random repair accuracy: {key_numbers['failure_conditioned_matched_random_accuracy']}",
        "",
        "## Scope and integrity",
        "",
        "- Controlled **regenerated** Waterbirds-style benchmark (CUB + Places); not the real",
        "  WILDS Waterbirds split unless those exact assets were supplied.",
        "- Finite, explicit candidate-intervention set; **not** open-world discovery; no exact",
        "  localization claim; no general-robustness claim.",
        "- Non-oracle CIC scoring used only model predicted distributions: no true label,",
        "  background label, correctness, or mask leakage.",
        "- Masks were used for dataset generation, oracle repair, evaluation, and post-hoc",
        "  candidate validation only.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
    ]

    examples_md = "# Regenerated Waterbirds Examples (first rows)\n\n" + _markdown_table(certs_nat.head(12))
    caption_md = (
        "# Caption\n\n"
        "Regenerated Waterbirds-style CIC pilot (CUB birds composited onto Places land/water "
        "backgrounds): aligned vs. misleading-background accuracy of pretrained CLIP, oracle "
        "background neutralization (upper bound from the known mask), CIC-selected intervention, "
        "and matched random background controls. Controlled regenerated benchmark; finite-candidate "
        "intervention; not open-world discovery.\n"
    )
    return _write_outputs(out_dir, cfg, metrics, certs_nat, certs_fc, key_numbers, summary, examples_md, caption_md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/regenerated_waterbirds_cic.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
