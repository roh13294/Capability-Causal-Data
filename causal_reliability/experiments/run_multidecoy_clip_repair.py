from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.data.clip_overlay_shortcuts import CLIP_OVERLAY_CLASSES
from causal_reliability.discovery.cic_region_scoring import neutralize_region
from causal_reliability.discovery.nonoracle_clip_discovery import discover_clip_shortcut_regions
from causal_reliability.experiments.run_clip_overlay_validation import _downloads_allowed
from causal_reliability.experiments.run_nonoracle_clip_repair import (
    PROMPT_TEMPLATE,
    _consensus_repair,
    _device,
    _iou,
    _predict_pil,
)
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


NEUTRAL_WORDS = ["shape", "object", "image", "sample", "figure"]
IRRELEVANT_WORDS = ["paper", "mark", "visual", "panel", "note"]
REGIMES = ["multi_decoy_misleading", "multi_decoy_aligned", "multi_decoy_neutral", "no_overlay"]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", max(10, size // 10))
    except Exception:
        return ImageFont.load_default()


def _shape_points(label: int, size: int) -> list[tuple[float, float]]:
    cx, cy, r = size * 0.5, size * 0.48, size * 0.21
    if label == 2:
        return [(cx, cy - 1.15 * r), (cx - 1.15 * r, cy + r), (cx + 1.15 * r, cy + r)]
    points = []
    for k in range(10):
        rad = r if k % 2 == 0 else r * 0.45
        ang = -np.pi / 2 + k * np.pi / 5
        points.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return points


def _draw_shape(draw: ImageDraw.ImageDraw, label: int, size: int) -> tuple[int, int, int, int]:
    box = [size * 0.29, size * 0.22, size * 0.71, size * 0.64]
    fill = (32, 34, 36)
    if label == 0:
        draw.ellipse(box, fill=fill)
    elif label == 1:
        draw.rectangle(box, fill=fill)
    else:
        draw.polygon(_shape_points(label, size), fill=fill)
    return tuple(int(v) for v in box)


def _box_slots(size: int) -> list[tuple[int, int]]:
    return [
        (int(size * 0.07), int(size * 0.06)),
        (int(size * 0.58), int(size * 0.07)),
        (int(size * 0.06), int(size * 0.72)),
        (int(size * 0.58), int(size * 0.72)),
        (int(size * 0.34), int(size * 0.86)),
    ]


def render_multidecoy_image(
    label: int,
    regime: str,
    index: int,
    *,
    size: int = 224,
    n_text_boxes: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(10_000 + label * 997 + index * 37 + REGIMES.index(regime) * 101)
    class_names = CLIP_OVERLAY_CLASSES
    img = Image.new("RGB", (size, size), (238, 240, 235))
    draw = ImageDraw.Draw(img)
    object_bbox = _draw_shape(draw, label, size)
    font = _font(size)
    slots = _box_slots(size)[:n_text_boxes]
    order = list(rng.permutation(len(slots)))
    words: list[tuple[str, str]] = []
    harmful_text = ""
    harmful_box_id = -1
    if regime == "multi_decoy_misleading":
        wrong = (label + 1 + index % (len(class_names) - 1)) % len(class_names)
        harmful_text = class_names[wrong]
        harmful_box_id = int(order[0])
        words.append((harmful_text, "harmful"))
    elif regime == "multi_decoy_aligned":
        harmful_text = class_names[label]
        harmful_box_id = int(order[0])
        words.append((harmful_text, "aligned"))
    if regime != "no_overlay":
        pool = NEUTRAL_WORDS + IRRELEVANT_WORDS
        k = n_text_boxes - len(words)
        words.extend((pool[(index + j + label) % len(pool)], "decoy") for j in range(k))

    text_boxes: list[dict[str, Any]] = []
    fixed_w = int(size * 0.34)
    fixed_h = int(size * 0.105)
    pad = max(3, size // 55)
    for slot_idx, (word, role) in zip(order, words):
        sx, sy = slots[slot_idx]
        x0, y0 = sx, sy
        x1, y1 = min(size - 3, sx + fixed_w), min(size - 3, sy + fixed_h)
        tb = draw.textbbox((0, 0), word, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = x0 + max(pad, (x1 - x0 - tw) // 2)
        ty = y0 + max(pad, (y1 - y0 - th) // 2)
        bbox = [int(x0), int(y0), int(x1), int(y1)]
        draw.rounded_rectangle(bbox, radius=3, fill=(255, 255, 255))
        draw.text((tx, ty), word, font=font, fill=(180, 20, 24))
        text_boxes.append({"box_id": int(slot_idx), "text": word, "role": role, "bbox": bbox})

    decoys = [b["bbox"] for b in text_boxes if b["role"] == "decoy"]
    harmful_bbox = next((b["bbox"] for b in text_boxes if b["role"] in {"harmful", "aligned"}), [])
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, {
        "true_label": class_names[label],
        "label": label,
        "harmful_text": harmful_text,
        "harmful_bbox": harmful_bbox,
        "decoy_bboxes": decoys,
        "all_text_boxes": text_boxes,
        "harmful_box_id": harmful_box_id,
        "object_bbox": list(object_bbox),
    }


def make_multidecoy_dataset(
    n_per_class: int,
    *,
    size: int,
    regimes: list[str],
    split: str,
    start_id: int = 0,
    n_text_boxes: int = 5,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    eid = start_id
    for regime in regimes:
        for label, class_name in enumerate(CLIP_OVERLAY_CLASSES):
            for j in range(n_per_class):
                image, meta = render_multidecoy_image(label, regime, j, size=size, n_text_boxes=n_text_boxes)
                examples.append(
                    {
                        "example_id": eid,
                        "split": split,
                        "regime": regime,
                        "label": label,
                        "true_label": class_name,
                        "image": image,
                        **meta,
                    }
                )
                eid += 1
    return examples


def _augment_prediction(model: ClipZeroShotClassifier, ex: dict[str, Any], rng: np.random.Generator, n_views: int) -> np.ndarray:
    base = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
    w, h = base.size
    views = []
    for _ in range(n_views):
        img = ImageEnhance.Brightness(base).enhance(float(rng.uniform(0.84, 1.16)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.84, 1.16)))
        if rng.random() < 0.35:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.0))))
        crop = int(rng.integers(0, max(1, w // 20)))
        if crop:
            img = img.crop((crop, crop, w - crop, h - crop)).resize((w, h), Image.Resampling.BICUBIC)
        views.append(img)
    return _predict_pil(model, views).mean(axis=0)


class _PredictionCache:
    def __init__(self, model: ClipZeroShotClassifier) -> None:
        self.model = model
        self.cache: dict[bytes, np.ndarray] = {}

    @staticmethod
    def _key(image: Image.Image) -> bytes:
        rgb = image.convert("RGB")
        return rgb.size[0].to_bytes(4, "little") + rgb.size[1].to_bytes(4, "little") + rgb.tobytes()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        keys = [self._key(img) for img in images]
        missing_images: list[Image.Image] = []
        missing_keys: list[bytes] = []
        for img, key in zip(images, keys):
            if key not in self.cache:
                missing_images.append(img)
                missing_keys.append(key)
        if missing_images:
            probs = _predict_pil(self.model, missing_images)
            for key, row in zip(missing_keys, probs):
                self.cache[key] = np.asarray(row, dtype=np.float32)
        return np.stack([self.cache[key] for key in keys])


def _row(
    ex: dict[str, Any],
    method: str,
    class_names: list[str],
    original_probs: np.ndarray,
    repaired_probs: np.ndarray,
    selected_bbox: tuple[int, int, int, int] | None,
    selected_candidate_id: str,
    selected_proposal_type: str,
    *,
    abstained: bool = False,
    oracle: bool = False,
    repair_action: str = "repair",
) -> dict[str, Any]:
    orig_pred = int(original_probs.argmax())
    rep_pred = None if abstained else int(repaired_probs.argmax())
    eps = 1e-12
    orig_safe = np.clip(np.asarray(original_probs, dtype=float), eps, 1.0)
    rep_safe = np.clip(np.asarray(repaired_probs, dtype=float), eps, 1.0)
    orig_safe = orig_safe / orig_safe.sum()
    rep_safe = rep_safe / rep_safe.sum()
    midpoint = 0.5 * (orig_safe + rep_safe)
    harmful_bbox = ex.get("harmful_bbox") or []
    selected_harmful_iou = 0.0 if selected_bbox is None or not harmful_bbox else _iou(selected_bbox, harmful_bbox)
    selected_decoy_iou = 0.0 if selected_bbox is None else max([_iou(selected_bbox, b) for b in ex.get("decoy_bboxes", [])] or [0.0])
    selected_object_iou = 0.0 if selected_bbox is None else _iou(selected_bbox, ex.get("object_bbox", [0, 0, 0, 0]))
    return {
        "example_id": ex["example_id"],
        "split": ex["split"],
        "regime": ex["regime"],
        "true_label": ex["true_label"],
        "label": int(ex["label"]),
        "method": method,
        "original_prediction": class_names[orig_pred],
        "original_prediction_index": orig_pred,
        "original_confidence": float(original_probs.max()),
        "original_top_class_probability_before": float(original_probs[orig_pred]),
        "original_correct": bool(orig_pred == int(ex["label"])),
        "repaired_prediction": "" if rep_pred is None else class_names[rep_pred],
        "repaired_prediction_index": np.nan if rep_pred is None else rep_pred,
        "repaired_confidence": 0.0 if rep_pred is None else float(repaired_probs.max()),
        "original_top_class_probability_after": float(repaired_probs[orig_pred]),
        "drop_in_original_top_class_probability": float(original_probs[orig_pred] - repaired_probs[orig_pred]),
        "prediction_flipped": False if rep_pred is None else bool(rep_pred != orig_pred),
        "js_shift": float(
            0.5 * np.sum(orig_safe * np.log(orig_safe / midpoint))
            + 0.5 * np.sum(rep_safe * np.log(rep_safe / midpoint))
        ),
        "kl_shift": float(np.sum(orig_safe * np.log(orig_safe / rep_safe))),
        "repaired_correct": False if rep_pred is None else bool(rep_pred == int(ex["label"])),
        "selected_candidate_id": selected_candidate_id,
        "selected_proposal_type": selected_proposal_type,
        "selected_bbox": "" if selected_bbox is None else json.dumps([int(v) for v in selected_bbox]),
        "selected_harmful_iou": float(selected_harmful_iou),
        "selected_decoy_iou": float(selected_decoy_iou),
        "selected_object_iou": float(selected_object_iou),
        "oracle_upper_bound": bool(oracle),
        "repair_action": repair_action,
        "abstained": bool(abstained),
}


def _example_image_hash(ex: dict[str, Any]) -> str:
    arr = (np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _random_patch_like(ex: dict[str, Any], area_fraction: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = ex["image"].shape[:2]
    area = max(25, int(area_fraction * w * h))
    bh = max(8, int(np.sqrt(area / 2.8)))
    bw = max(10, int(area / bh))
    for _ in range(100):
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        box = (x0, y0, min(w, x0 + bw), min(h, y0 + bh))
        if max([_iou(box, b["bbox"]) for b in ex["all_text_boxes"]] or [0.0]) < 0.05:
            return box
    return (0, 0, min(w, bw), min(h, bh))


def _policy_action(original_probs: np.ndarray, top1_probs: np.ndarray, top1: Any | None, top3_probs: np.ndarray, stable: bool, policy: dict[str, Any], *, allow_abstain: bool) -> tuple[np.ndarray, bool, tuple[int, int, int, int] | None, str, str, str]:
    if top1 is None or float(top1.score) < float(policy.get("score_threshold", np.inf)):
        return original_probs, False, None, "none", "none", "keep_original"
    ok = bool(stable and float(top1.consensus_stability) >= float(policy.get("min_consensus_stability", 2 / 3)))
    if not ok:
        if allow_abstain:
            return original_probs, True, top1.bbox, top1.candidate_id, top1.proposal_type, "abstain"
        return original_probs, False, None, "none", "none", "keep_original"
    return top3_probs, False, top1.bbox, top1.candidate_id, top1.proposal_type, "repair"


def _evaluate_examples(
    *,
    examples: list[dict[str, Any]],
    class_names: list[str],
    prompts: list[str],
    model: ClipZeroShotClassifier,
    seed: int,
    max_candidates: int,
    n_views: int,
    rng: np.random.Generator,
    policy: dict[str, Any] | None,
    random_draws: int,
    cache_dir: Path | None = None,
    resume: bool = False,
    progress_label: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, Any]] = []
    cert_rows: list[dict[str, Any]] = []
    default_policy = {"score_threshold": float("inf"), "min_consensus_stability": 2 / 3}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            expected = []
            for ex in examples:
                cache_key = f"example_{int(ex['example_id'])}_{_example_image_hash(ex)}"
                expected.append((cache_dir / f"{cache_key}_certificates.csv", cache_dir / f"{cache_key}_rankings.csv"))
            if expected and all(cert.exists() and rank.exists() for cert, rank in expected):
                for index, (cert_path, rank_path) in enumerate(expected, start=1):
                    cert_rows.extend(pd.read_csv(cert_path).to_dict("records"))
                    ranking_rows.extend(pd.read_csv(rank_path).to_dict("records"))
                    if progress_label:
                        print(f"[{progress_label}] reused example {index}/{len(examples)}", flush=True)
                return pd.DataFrame(cert_rows), pd.DataFrame(ranking_rows)
    for index, ex in enumerate(examples, start=1):
        cache_key = f"example_{int(ex['example_id'])}_{_example_image_hash(ex)}"
        cert_path = cache_dir / f"{cache_key}_certificates.csv" if cache_dir is not None else None
        rank_path = cache_dir / f"{cache_key}_rankings.csv" if cache_dir is not None else None
        ex_ranking_rows: list[dict[str, Any]] = []
        ex_cert_rows: list[dict[str, Any]] = []
        pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        predict_fn = _PredictionCache(model)
        _, scores, original_probs = discover_clip_shortcut_regions(pil, predict_fn, prompts, seed=seed + int(ex["example_id"]), max_candidates=max_candidates)
        harmful_bbox = tuple(ex["harmful_bbox"]) if ex["harmful_bbox"] else None
        for rank, score in enumerate(scores, start=1):
            decoy_iou = max([_iou(score.bbox, b) for b in ex["decoy_bboxes"]] or [0.0])
            row = score.to_dict()
            row.update(
                {
                    "example_id": ex["example_id"],
                    "split": ex["split"],
                    "regime": ex["regime"],
                    "rank": rank,
                    "harmful_iou": 0.0 if harmful_bbox is None else _iou(score.bbox, harmful_bbox),
                    "decoy_iou": decoy_iou,
                    "object_iou": _iou(score.bbox, ex["object_bbox"]),
                    "harmful_bbox_eval_only": "" if harmful_bbox is None else json.dumps(list(harmful_bbox)),
                }
            )
            ex_ranking_rows.append(row)
        top1 = scores[0] if scores else None
        top1_probs = predict_fn([neutralize_region(pil, top1.bbox)])[0] if top1 else original_probs
        top3_probs, top3_stable, top3_bbox, top3_id, top3_type = _consensus_repair(pil, scores[:3], predict_fn)
        text_like = [s for s in scores if s.proposal_type in {"text_box_component", "textness_high_frequency", "horizontal_text_band", "corner_edge_watermark"}]
        largest = max(text_like, key=lambda s: s.area_fraction, default=top1)
        highest_textness = max(text_like, key=lambda s: s.textness_score, default=top1)
        largest_probs = predict_fn([neutralize_region(pil, largest.bbox)])[0] if largest else original_probs
        textness_probs = predict_fn([neutralize_region(pil, highest_textness.bbox)])[0] if highest_textness else original_probs
        aug_probs = _augment_prediction(model, ex, rng, n_views)
        oracle_probs = predict_fn([neutralize_region(pil, harmful_bbox)])[0] if harmful_bbox else original_probs
        selected_policy = policy or default_policy
        clean_probs, clean_abs, clean_bbox, clean_id, clean_type, clean_action = _policy_action(original_probs, top1_probs, top1, top3_probs, top3_stable, selected_policy, allow_abstain=False)
        selective_probs, selective_abs, selective_bbox, selective_id, selective_type, selective_action = _policy_action(original_probs, top1_probs, top1, top3_probs, top3_stable, selected_policy, allow_abstain=True)

        ex_cert_rows.extend(
            [
                _row(ex, "original_clip_prediction", class_names, original_probs, original_probs, None, "none", "none", repair_action="keep_original"),
                _row(ex, "oracle_harmful_text_neutralization", class_names, original_probs, oracle_probs, harmful_bbox, "oracle_harmful_bbox", "oracle upper bound", oracle=True),
                _row(ex, "nonoracle_cic_top1_region_repair", class_names, original_probs, top1_probs, top1.bbox if top1 else None, top1.candidate_id if top1 else "", top1.proposal_type if top1 else ""),
                _row(ex, "nonoracle_cic_top3_consensus_repair", class_names, original_probs, top3_probs, top3_bbox, top3_id, top3_type, abstained=not top3_stable, repair_action="repair" if top3_stable else "abstain"),
                _row(ex, "nonoracle_cic_clean_safe_repair", class_names, original_probs, clean_probs, clean_bbox, clean_id, clean_type, abstained=clean_abs, repair_action=clean_action),
                _row(ex, "nonoracle_cic_selective_repair_or_abstain", class_names, original_probs, selective_probs, selective_bbox, selective_id, selective_type, abstained=selective_abs, repair_action=selective_action),
                _row(ex, "largest_text_region_repair", class_names, original_probs, largest_probs, largest.bbox if largest else None, largest.candidate_id if largest else "", largest.proposal_type if largest else ""),
                _row(ex, "highest_textness_region_repair", class_names, original_probs, textness_probs, highest_textness.bbox if highest_textness else None, highest_textness.candidate_id if highest_textness else "", highest_textness.proposal_type if highest_textness else ""),
                _row(ex, "random_augmentation_consensus", class_names, original_probs, aug_probs, None, "random_augmentation", "random_augmentation"),
            ]
        )
        if text_like:
            random_text_choices = []
            for draw_id in range(random_draws):
                chosen = text_like[int(rng.integers(0, len(text_like)))]
                random_text_choices.append((draw_id, chosen))
            random_text_probs = predict_fn([neutralize_region(pil, chosen.bbox) for _, chosen in random_text_choices])
            for (draw_id, chosen), probs in zip(random_text_choices, random_text_probs):
                ex_cert_rows.append(_row(ex, "random_matched_text_region_repair", class_names, original_probs, probs, chosen.bbox, f"{chosen.candidate_id}:draw{draw_id}", chosen.proposal_type))
        target_area = float(top1.area_fraction) if top1 else 0.035
        random_nontext_boxes = []
        for draw_id in range(random_draws):
            box = _random_patch_like(ex, target_area, rng)
            random_nontext_boxes.append((draw_id, box))
        random_nontext_probs = predict_fn([neutralize_region(pil, box) for _, box in random_nontext_boxes])
        for (draw_id, box), probs in zip(random_nontext_boxes, random_nontext_probs):
            ex_cert_rows.append(_row(ex, "random_nontext_patch_repair", class_names, original_probs, probs, box, f"random_nontext_{draw_id}", "random_nontext_patch"))
        if cert_path is not None and rank_path is not None:
            pd.DataFrame(ex_cert_rows).to_csv(cert_path, index=False)
            pd.DataFrame(ex_ranking_rows).to_csv(rank_path, index=False)
        cert_rows.extend(ex_cert_rows)
        ranking_rows.extend(ex_ranking_rows)
        if progress_label:
            print(f"[{progress_label}] finished example {index}/{len(examples)} id={ex['example_id']}", flush=True)
    return pd.DataFrame(cert_rows), pd.DataFrame(ranking_rows)


def _method_metrics(certs: pd.DataFrame, method: str, status: ClipStatus) -> dict[str, Any]:
    df = certs[certs["method"] == method]
    non_abs = ~df["abstained"].astype(bool)
    original = df["original_correct"].astype(bool)
    repaired = df["repaired_correct"].astype(bool)
    row: dict[str, Any] = {
        "method": method,
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        "n_examples": int(len(df)),
        "n_non_abstained": int(non_abs.sum()),
        "coverage": float(non_abs.mean()) if len(df) else np.nan,
        "abstention_rate": float((~non_abs).mean()) if len(df) else np.nan,
        "original_accuracy": float(original.mean()) if len(df) else np.nan,
        "repaired_accuracy": float(repaired[non_abs].mean()) if bool(non_abs.sum()) else np.nan,
        "accuracy_counting_abstentions_wrong": float(repaired.mean()) if len(df) else np.nan,
        "selective_accuracy": float(repaired[non_abs].mean()) if bool(non_abs.sum()) else np.nan,
        "high_confidence_failure_rate_before": float(((~original) & (df["original_confidence"] >= 0.80)).mean()) if len(df) else np.nan,
        "high_confidence_failure_rate_after": float(((~repaired) & (df["repaired_confidence"] >= 0.80) & non_abs).mean()) if len(df) else np.nan,
        "repair_success_rate": float(((~original) & repaired & non_abs).sum() / max(1, int((~original).sum()))) if len(df) else np.nan,
        "failure_capture_rate": float(((~non_abs) & (~original)).sum() / max(1, int((~original).sum()))) if len(df) else np.nan,
        "false_abstention_rate": float(((~non_abs) & original).sum() / max(1, int(original.sum()))) if len(df) else np.nan,
        "selected_harmful_localization_iou_0_3": float((df["selected_harmful_iou"] >= 0.3).mean()) if len(df) and "selected_harmful_iou" in df else np.nan,
        "selected_harmful_localization_iou_0_5": float((df["selected_harmful_iou"] >= 0.5).mean()) if len(df) and "selected_harmful_iou" in df else np.nan,
        "selected_decoy_overlap_rate_iou_0_3": float((df["selected_decoy_iou"] >= 0.3).mean()) if len(df) and "selected_decoy_iou" in df else np.nan,
        "selected_object_overlap_rate_iou_0_2": float((df["selected_object_iou"] >= 0.2).mean()) if len(df) and "selected_object_iou" in df else np.nan,
    }
    for regime in REGIMES:
        sub = df[df["regime"] == regime]
        sub_non_abs = ~sub["abstained"].astype(bool) if len(sub) else pd.Series(dtype=bool)
        key = regime.replace("multi_decoy_", "")
        row[f"{key}_accuracy_before"] = float(sub["original_correct"].astype(bool).mean()) if len(sub) else np.nan
        row[f"{key}_accuracy_after"] = float(sub.loc[sub_non_abs, "repaired_correct"].astype(bool).mean()) if len(sub) and bool(sub_non_abs.sum()) else np.nan
    clean_before = np.nanmean([row.get("aligned_accuracy_before", np.nan), row.get("neutral_accuracy_before", np.nan), row.get("no_overlay_accuracy_before", np.nan)])
    clean_after = np.nanmean([row.get("aligned_accuracy_after", np.nan), row.get("neutral_accuracy_after", np.nan), row.get("no_overlay_accuracy_after", np.nan)])
    row["clean_accuracy_drop"] = clean_before - clean_after if np.isfinite(clean_before) and np.isfinite(clean_after) else np.nan
    if method in {"random_matched_text_region_repair", "random_nontext_patch_repair"} and len(df):
        draw = df["selected_candidate_id"].astype(str).str.extract(r"(?:draw|random_nontext_)(\d+)")[0]
        draw = draw.fillna("0")
        per_draw = []
        per_draw_loc = []
        for _, sub in df.assign(_draw=draw).groupby("_draw"):
            sub_non_abs = ~sub["abstained"].astype(bool)
            if bool(sub_non_abs.sum()):
                per_draw.append(float(sub.loc[sub_non_abs, "repaired_correct"].astype(bool).mean()))
            if "selected_harmful_iou" in sub:
                per_draw_loc.append(float((sub["selected_harmful_iou"] >= 0.3).mean()))
        if per_draw:
            arr = np.asarray(per_draw, dtype=float)
            row["random_draw_repaired_accuracy_mean"] = float(arr.mean())
            row["random_draw_repaired_accuracy_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            row["random_draw_repaired_accuracy_ci95"] = float(1.96 * row["random_draw_repaired_accuracy_std"] / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        if per_draw_loc:
            arr = np.asarray(per_draw_loc, dtype=float)
            row["random_draw_localization_iou_0_3_mean"] = float(arr.mean())
            row["random_draw_localization_iou_0_3_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            row["random_draw_localization_iou_0_3_ci95"] = float(1.96 * row["random_draw_localization_iou_0_3_std"] / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return row


def _metrics(certs: pd.DataFrame, rankings: pd.DataFrame, status: ClipStatus) -> pd.DataFrame:
    rows = [_method_metrics(certs, method, status) for method in certs["method"].drop_duplicates()]
    out = pd.DataFrame(rows)
    misleading = rankings[rankings["regime"] == "multi_decoy_misleading"]
    if len(misleading):
        groups = [g for _, g in misleading.groupby("example_id")]
        top1 = misleading[misleading["rank"] == 1]
        loc = {
            "harmful_top1_iou_0_3": float((top1["harmful_iou"] >= 0.3).mean()) if len(top1) else np.nan,
            "harmful_top1_iou_0_5": float((top1["harmful_iou"] >= 0.5).mean()) if len(top1) else np.nan,
            "harmful_top3_iou_0_3": float(np.mean([(g.nsmallest(3, "rank")["harmful_iou"] >= 0.3).any() for g in groups])),
            "harmful_top3_iou_0_5": float(np.mean([(g.nsmallest(3, "rank")["harmful_iou"] >= 0.5).any() for g in groups])),
            "median_harmful_region_rank": float(np.median([g.sort_values("harmful_iou", ascending=False)["rank"].iloc[0] for g in groups])),
            "mean_harmful_region_rank": float(np.mean([g.sort_values("harmful_iou", ascending=False)["rank"].iloc[0] for g in groups])),
            "top_candidate_overlaps_harmful": float((top1["harmful_iou"] >= 0.3).mean()) if len(top1) else np.nan,
            "top_candidate_overlaps_neutral_decoy": float((top1["decoy_iou"] >= 0.3).mean()) if len(top1) else np.nan,
            "top_candidate_overlaps_object_center": float((top1["object_iou"] >= 0.2).mean()) if len(top1) else np.nan,
        }
        for k, v in loc.items():
            out[k] = v
    return out


def _policy_metrics(certs: pd.DataFrame, rankings: pd.DataFrame, threshold: float, min_consensus: float) -> dict[str, Any]:
    base = certs[certs["method"] == "nonoracle_cic_top3_consensus_repair"]
    top1 = rankings[rankings["rank"] == 1].set_index("example_id")
    rows = []
    for _, row in base.iterrows():
        orig = certs[(certs["example_id"] == row["example_id"]) & (certs["method"] == "original_clip_prediction")].iloc[0].copy()
        score = top1.loc[row["example_id"]] if row["example_id"] in top1.index else None
        action = "keep_original"
        repaired = bool(orig["original_correct"])
        abstained = False
        if score is not None and float(score["score"]) >= threshold:
            if not bool(row["abstained"]) and float(score["consensus_stability"]) >= min_consensus:
                action = "repair"
                repaired = bool(row["repaired_correct"])
        orig["method"] = "policy_eval"
        orig["repair_action"] = action
        orig["repaired_correct"] = repaired
        orig["abstained"] = abstained
        rows.append(orig)
    df = pd.DataFrame(rows)
    clean = df["regime"].isin(["multi_decoy_aligned", "multi_decoy_neutral", "no_overlay"])
    misleading = df["regime"] == "multi_decoy_misleading"
    before = float(df.loc[clean, "original_correct"].astype(bool).mean()) if bool(clean.sum()) else np.nan
    after = float(df.loc[clean, "repaired_correct"].astype(bool).mean()) if bool(clean.sum()) else np.nan
    return {
        "score_threshold": float(threshold),
        "min_consensus_stability": float(min_consensus),
        "misleading_repair_accuracy": float(df.loc[misleading, "repaired_correct"].astype(bool).mean()) if bool(misleading.sum()) else np.nan,
        "clean_accuracy_drop": before - after if np.isfinite(before) and np.isfinite(after) else np.nan,
        "false_repair_rate": float((df.loc[clean, "repair_action"] == "repair").mean()) if bool(clean.sum()) else np.nan,
        "coverage": 1.0,
    }


def _select_policy(certs: pd.DataFrame, rankings: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    thresholds = sorted(set([0.0, *rankings.loc[rankings["rank"] == 1, "score"].astype(float).tolist()]))
    consensus_grid = [float(v) for v in cfg.get("policy", {}).get("min_consensus_grid", [2 / 3, 1.0])]
    rows = [_policy_metrics(certs, rankings, t, c) for c in consensus_grid for t in thresholds]
    sweep = pd.DataFrame(rows)
    max_drop = float(cfg.get("policy", {}).get("max_clean_drop", 0.05))
    safe = sweep[sweep["clean_accuracy_drop"] <= max_drop]
    if safe.empty:
        safe = sweep
    chosen = safe.sort_values(["misleading_repair_accuracy", "clean_accuracy_drop", "false_repair_rate"], ascending=[False, True, True]).iloc[0]
    return {
        "objective": "validation_clean_safe_repair",
        "score_threshold": float(chosen["score_threshold"]),
        "min_consensus_stability": float(chosen["min_consensus_stability"]),
        "max_clean_drop": max_drop,
        "validation_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in chosen.items()},
    }, sweep


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus) -> dict[str, str]:
    pd.DataFrame().to_csv(out_dir / "multidecoy_candidate_rankings.csv", index=False)
    certs = pd.DataFrame([{"method": "unavailable", "oracle_upper_bound": False, "pretrained_loaded": False}])
    metrics = pd.DataFrame([{"method": "unavailable", "evidence_status": "unavailable", "headline_eligible": False, "backend": status.backend, "model_name": status.model_name, "pretrained_loaded": False, "headline_eligibility_reasons": status.error_message or "pretrained CLIP did not load"}])
    certs.to_csv(out_dir / "multidecoy_repair_certificates.csv", index=False)
    metrics.to_csv(out_dir / "multidecoy_repair_metrics.csv", index=False)
    (out_dir / "multidecoy_repair_summary.md").write_text("# Multi-Decoy CLIP Repair\n\nPretrained CLIP unavailable; no fake headline evidence was generated.\n", encoding="utf-8")
    (out_dir / "multidecoy_repair_examples.md").write_text("# Multi-Decoy Examples\n\nUnavailable.\n", encoding="utf-8")
    (out_dir / "multidecoy_repair_caption.md").write_text("# Caption\n\nUnavailable.\n", encoding="utf-8")
    (out_dir / "selected_multidecoy_repair_policy.json").write_text(json.dumps({"unavailable": True}, indent=2), encoding="utf-8")
    pd.DataFrame().to_csv(out_dir / "validation_policy_sweep.csv", index=False)
    _plot(metrics, out_dir / "multidecoy_repair_plot.png", out_dir / "multidecoy_repair_plot.pdf")
    return {"metrics": str(out_dir / "multidecoy_repair_metrics.csv"), "certificates": str(out_dir / "multidecoy_repair_certificates.csv"), "rankings": str(out_dir / "multidecoy_candidate_rankings.csv"), "summary": str(out_dir / "multidecoy_repair_summary.md")}


def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    take = metrics[metrics.get("method", pd.Series(dtype=str)).isin(["original_clip_prediction", "oracle_harmful_text_neutralization", "nonoracle_cic_top1_region_repair", "nonoracle_cic_top3_consensus_repair", "random_matched_text_region_repair", "largest_text_region_repair", "highest_textness_region_repair"])] if len(metrics) else pd.DataFrame()
    plt.figure(figsize=(9.6, 4.8))
    if len(take) and "misleading_accuracy_after" in take:
        x = np.arange(len(take))
        plt.bar(x, take["misleading_accuracy_after"], color="#4c78a8")
        plt.xticks(x, take["method"], rotation=25, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("Multi-decoy misleading accuracy after")
    else:
        plt.text(0.5, 0.5, "No eligible multi-decoy CLIP metrics", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "multidecoy_clip_repair")
    data_cfg = cfg.get("data", {})
    regimes = list(data_cfg.get("regimes", REGIMES))
    size = int(data_cfg.get("image_size", 224))
    n_text_boxes = int(data_cfg.get("n_text_boxes", 5))
    test_n = int(data_cfg.get("test_n_per_class", data_cfg.get("n_per_class", 8)))
    val_n = int(data_cfg.get("validation_n_per_class", max(2, test_n)))
    val_examples = make_multidecoy_dataset(val_n, size=size, regimes=regimes, split="validation", start_id=0, n_text_boxes=n_text_boxes)
    test_examples = make_multidecoy_dataset(test_n, size=size, regimes=regimes, split="test", start_id=len(val_examples), n_text_boxes=n_text_boxes)

    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_clip_multidecoy", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for CLIP repair evidence")
        return _write_unavailable(out_dir, cfg, status)
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, cfg, status)

    prompts = [PROMPT_TEMPLATE.format(label=name) for name in CLIP_OVERLAY_CLASSES]
    model = ClipZeroShotClassifier(status, CLIP_OVERLAY_CLASSES.copy(), prompts=prompts, device=device)
    max_candidates = int(cfg.get("max_candidates", 96))
    n_views = int(cfg.get("augmentation_views", 5))
    random_draws = int(cfg.get("random_draws", 100))
    val_certs, val_rankings = _evaluate_examples(examples=val_examples, class_names=CLIP_OVERLAY_CLASSES.copy(), prompts=prompts, model=model, seed=seed + 100_000, max_candidates=max_candidates, n_views=n_views, rng=rng, policy=None, random_draws=max(1, min(5, random_draws)))
    policy, sweep = _select_policy(val_certs, val_rankings, cfg)
    (out_dir / "selected_multidecoy_repair_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    sweep.to_csv(out_dir / "validation_policy_sweep.csv", index=False)
    certs, rankings = _evaluate_examples(examples=test_examples, class_names=CLIP_OVERLAY_CLASSES.copy(), prompts=prompts, model=model, seed=seed, max_candidates=max_candidates, n_views=n_views, rng=rng, policy=policy, random_draws=random_draws)
    metrics = _metrics(certs, rankings, status)
    rankings.to_csv(out_dir / "multidecoy_candidate_rankings.csv", index=False)
    certs.to_csv(out_dir / "multidecoy_repair_certificates.csv", index=False)
    metrics.to_csv(out_dir / "multidecoy_repair_metrics.csv", index=False)
    (out_dir / "multidecoy_clip_repair_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot(metrics, out_dir / "multidecoy_repair_plot.png", out_dir / "multidecoy_repair_plot.pdf")
    lookup = metrics.set_index("method").to_dict("index")
    top1 = lookup.get("nonoracle_cic_top1_region_repair", {})
    random_text = lookup.get("random_matched_text_region_repair", {})
    summary = [
        "# Multi-Decoy CLIP Shortcut Localization and Repair",
        "",
        "This benchmark renders multiple visually similar text regions per image. Only evaluation/oracle code sees `harmful_bbox`, `harmful_text`, or decoy metadata; non-oracle proposals and CIC scoring receive image pixels and CLIP predictions only.",
        "",
        f"Backend: {status.backend}. Model: {status.model_name}. Pretrained loaded: `{status.pretrained}`.",
        f"Validation-selected policy: threshold {policy.get('score_threshold'):.6f}, min consensus {policy.get('min_consensus_stability'):.3f}.",
        "",
        "## Results",
        "",
        f"- CIC top-1 misleading accuracy after repair: {top1.get('misleading_accuracy_after', np.nan):.3f}",
        f"- Random matched text-region misleading accuracy after repair: {random_text.get('misleading_accuracy_after', np.nan):.3f}",
        f"- CIC top-1/top-3 harmful localization at IoU >= 0.3: {top1.get('harmful_top1_iou_0_3', np.nan):.3f} / {top1.get('harmful_top3_iou_0_3', np.nan):.3f}",
        "",
        "Do not treat the oracle upper bound as automatic shortcut discovery evidence. Do not headline this result unless the held-out metrics show CIC clearly beating matched random text-region masking.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
    ]
    (out_dir / "multidecoy_repair_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (out_dir / "multidecoy_repair_examples.md").write_text("# Multi-Decoy Repair Examples\n\n" + _markdown_table(certs.head(20)), encoding="utf-8")
    (out_dir / "multidecoy_repair_caption.md").write_text("# Multi-Decoy CLIP Repair Figure Caption\n\nHeld-out multi-decoy misleading-overlay repair accuracy for original CLIP, oracle harmful-text neutralization, non-oracle CIC region repair, and matched random/control baselines.\n", encoding="utf-8")
    return {"metrics": str(out_dir / "multidecoy_repair_metrics.csv"), "certificates": str(out_dir / "multidecoy_repair_certificates.csv"), "rankings": str(out_dir / "multidecoy_candidate_rankings.csv"), "summary": str(out_dir / "multidecoy_repair_summary.md"), "selected_policy": str(out_dir / "selected_multidecoy_repair_policy.json"), "validation_policy_sweep": str(out_dir / "validation_policy_sweep.csv")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multidecoy_clip_repair.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
