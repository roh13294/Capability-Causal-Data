"""Cross-shortcut generalization test for the frozen CIC repair policy.

This experiment asks a single, narrow question:

    Can a CIC repair/scoring policy that was *selected on text-overlay shortcut
    failures* transfer, with no retuning, to a different finite-candidate
    shortcut family -- a non-text colored-symbol watermark?

The frozen policy (score threshold + consensus-stability threshold) is loaded
verbatim from ``results/hard_multidecoy_clip_repair/selected_repair_policy.json``.
It is NOT reselected, retuned, or reweighted on the cross-shortcut benchmark.
Region proposals are generic (sliding windows, high-frequency connected
components, corner/edge patches, contrast/edge-density regions, random matched
patches, center-object controls) and the non-oracle scorer receives only image
pixels, CLIP predictions, class prompts, and candidate proposals -- never the
true label, the harmful shortcut bbox, the shortcut type, or correctness.

True labels and the harmful shortcut bbox are used ONLY to (a) define the
held-out failure-conditioned subset and (b) compute the oracle upper bound and
localization metrics for reporting.

This is explicitly a *finite-candidate* transfer test. It is NOT open-world
shortcut discovery and it does NOT claim general robustness. If the frozen
policy fails to transfer, the script reports that honestly and is not headline
eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw

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
from causal_reliability.experiments.run_hard_multidecoy_clip_repair import _ci_text, _wilson_ci
from causal_reliability.experiments.run_multidecoy_clip_repair import (
    _PredictionCache,
    _augment_prediction,
    _example_image_hash,
    _random_patch_like,
    _row,
)
from causal_reliability.experiments.run_nonoracle_clip_repair import (
    PROMPT_TEMPLATE,
    _consensus_repair,
    _device,
    _iou,
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


# Non-text shortcut family used for this transfer test (Option A: colored symbol
# watermark badge). The watermark contains a small colored geometric symbol, not
# any readable text. Visually distinct from the text-overlay benchmark.
SHORTCUT_TYPE = "colored_symbol_watermark"

MIS_REGIME = "cross_shortcut_misleading"
ALIGNED_REGIME = "cross_shortcut_aligned"
NEUTRAL_REGIME = "cross_shortcut_neutral"
NO_OVERLAY_REGIME = "no_overlay"
CROSS_REGIMES = [MIS_REGIME, ALIGNED_REGIME, NEUTRAL_REGIME, NO_OVERLAY_REGIME]

# Vivid badge colors keyed to nothing semantic; the *symbol* carries the
# spurious class association, not the color alone.
BADGE_COLORS = [(214, 40, 40), (24, 90, 210), (28, 158, 90), (230, 150, 20)]
NEUTRAL_GLYPHS = ["dot", "ring", "plus", "bars"]
FROZEN_POLICY_LABEL = "Frozen text-selected CIC policy applied to non-text shortcut proposals."


# ---------------------------------------------------------------------------
# Rendering: centered shape + non-text colored-symbol watermark badges
# ---------------------------------------------------------------------------
def _shape_points(label: int, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    if label == 2:  # triangle
        return [(cx, cy - 1.15 * r), (cx - 1.15 * r, cy + r), (cx + 1.15 * r, cy + r)]
    points = []
    for k in range(10):
        rad = r if k % 2 == 0 else r * 0.45
        ang = -np.pi / 2 + k * np.pi / 5
        points.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return points


def _draw_class_shape_in_box(draw: ImageDraw.ImageDraw, label: int, box: tuple[float, float, float, float], fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    if label == 0:  # circle
        draw.ellipse([x0, y0, x1, y1], fill=fill)
    elif label == 1:  # square
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    else:  # triangle (2) / star (3)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        r = 0.5 * min(x1 - x0, y1 - y0)
        draw.polygon(_shape_points(label, cx, cy, r), fill=fill)


def _draw_neutral_glyph(draw: ImageDraw.ImageDraw, glyph: str, box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = 0.32 * min(x1 - x0, y1 - y0)
    if glyph == "dot":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif glyph == "ring":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=max(2, int(r * 0.35)))
    elif glyph == "plus":
        t = max(2, int(r * 0.4))
        draw.rectangle([cx - r, cy - t, cx + r, cy + t], fill=fill)
        draw.rectangle([cx - t, cy - r, cx + t, cy + r], fill=fill)
    else:  # bars (non-text horizontal bars; not letters)
        for j in range(3):
            yy = cy - r + j * r
            draw.rectangle([cx - r, yy, cx + r, yy + max(2, int(r * 0.3))], fill=fill)


def _badge_slots(size: int, badge_size: int) -> list[tuple[int, int]]:
    # Corner badge anchors placed away from / partially overlapping the centered
    # object. Anchors are chosen so a large badge stays inside the image.
    far = size - badge_size - int(size * 0.03)
    near = int(size * 0.03)
    return [
        (near, near),
        (far, near),
        (near, far),
        (far, far),
    ]


def _draw_badge(
    img: Image.Image,
    anchor: tuple[int, int],
    badge_size: int,
    color: tuple[int, int, int],
    *,
    class_symbol: int | None,
    glyph: str,
) -> list[int]:
    draw = ImageDraw.Draw(img)
    x0, y0 = anchor
    x1, y1 = x0 + badge_size, y0 + badge_size
    draw.rounded_rectangle([x0, y0, x1, y1], radius=max(3, badge_size // 8), fill=color)
    pad = max(2, badge_size // 12)
    inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    if class_symbol is not None:
        _draw_class_shape_in_box(draw, class_symbol, inner, (250, 250, 250))
    else:
        _draw_neutral_glyph(draw, glyph, tuple(int(v) for v in inner), (250, 250, 250))
    return [int(x0), int(y0), int(x1), int(y1)]


def render_cross_shortcut_image(
    label: int,
    regime: str,
    index: int,
    *,
    size: int = 200,
    benchmark_seed: int = 0,
    n_badges: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a centered class shape plus non-text colored-symbol watermark badges.

    misleading -> one badge carries a mini-shape of a WRONG class.
    aligned    -> one badge carries a mini-shape of the TRUE class.
    neutral    -> all badges carry neutral (non-class) glyphs.
    no_overlay -> just the centered object.
    """

    rng = np.random.default_rng(7_000_003 + benchmark_seed * 100_003 + label * 9_973 + index * 131 + CROSS_REGIMES.index(regime) * 17)
    n_classes = len(CLIP_OVERLAY_CLASSES)
    img = Image.new("RGB", (size, size), (238, 240, 235))
    draw = ImageDraw.Draw(img)
    obj_box = (size * 0.30, size * 0.24, size * 0.70, size * 0.64)
    _draw_class_shape_in_box(draw, label, obj_box, (30, 31, 34))
    object_bbox = [int(v) for v in obj_box]

    badge_size = int(size * 0.30)
    slots = _badge_slots(size, badge_size)
    order = list(rng.permutation(len(slots)))[: min(n_badges, len(slots))]

    harmful_bbox: list[int] = []
    decoy_bboxes: list[list[int]] = []
    all_badge_boxes: list[dict[str, Any]] = []
    shortcut_association = "none"

    if regime == NO_OVERLAY_REGIME:
        chosen_slots: list[int] = []
    else:
        chosen_slots = order

    for rank, slot_idx in enumerate(chosen_slots):
        anchor = slots[slot_idx]
        color = BADGE_COLORS[(index + slot_idx) % len(BADGE_COLORS)]
        is_shortcut = rank == 0 and regime in {MIS_REGIME, ALIGNED_REGIME}
        if is_shortcut:
            if regime == MIS_REGIME:
                wrong = (label + 1 + index % (n_classes - 1)) % n_classes
                class_symbol = wrong
                shortcut_association = CLIP_OVERLAY_CLASSES[wrong]
            else:
                class_symbol = label
                shortcut_association = CLIP_OVERLAY_CLASSES[label]
            box = _draw_badge(img, anchor, badge_size, color, class_symbol=class_symbol, glyph="dot")
            harmful_bbox = box
            all_badge_boxes.append({"box_id": int(slot_idx), "role": "harmful", "bbox": box})
        else:
            glyph = NEUTRAL_GLYPHS[(index + rank + slot_idx) % len(NEUTRAL_GLYPHS)]
            box = _draw_badge(img, anchor, badge_size, color, class_symbol=None, glyph=glyph)
            decoy_bboxes.append(box)
            all_badge_boxes.append({"box_id": int(slot_idx), "role": "decoy", "bbox": box})

    arr = np.asarray(img).astype(np.float32) / 255.0
    image_hash = hashlib.sha256((arr.clip(0, 1) * 255).astype(np.uint8).tobytes()).hexdigest()[:16]
    meta = {
        "true_label": CLIP_OVERLAY_CLASSES[label],
        "label": int(label),
        "shortcut_type": SHORTCUT_TYPE,
        "shortcut_label_association": shortcut_association,
        "harmful_shortcut_bbox": harmful_bbox,
        # Aliases so the shared non-text repair engine / oracle code can reuse keys.
        "harmful_bbox": harmful_bbox,
        "harmful_text": "",
        "decoy_bboxes": decoy_bboxes,
        "all_text_boxes": all_badge_boxes,
        "object_bbox": object_bbox,
        "seed": int(benchmark_seed),
        "image_hash": image_hash,
    }
    return arr, meta


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.asarray(arr).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


# ---------------------------------------------------------------------------
# Repair + baselines on non-text proposals using the FROZEN text-selected policy
# ---------------------------------------------------------------------------
SHORTCUT_LIKE = {"text_box_component", "textness_high_frequency", "horizontal_text_band", "corner_edge_watermark", "sliding_small", "sliding_medium"}


def _frozen_policy(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return None if data.get("unavailable") else data


def _policy_action(original_probs, top1_probs, top1, top3_probs, stable, policy, *, allow_abstain):
    """Apply the FROZEN policy (threshold + consensus). No tuning happens here."""
    if top1 is None or float(top1.score) < float(policy.get("score_threshold", np.inf)):
        return original_probs, False, None, "none", "none", "keep_original"
    ok = bool(stable and float(top1.consensus_stability) >= float(policy.get("min_consensus_stability", 2 / 3)))
    if not ok:
        if allow_abstain:
            return original_probs, True, top1.bbox, top1.candidate_id, top1.proposal_type, "abstain"
        return original_probs, False, None, "none", "none", "keep_original"
    return top3_probs, False, top1.bbox, top1.candidate_id, top1.proposal_type, "repair"


def evaluate_cross_shortcut(
    *,
    examples: list[dict[str, Any]],
    model: ClipZeroShotClassifier,
    prompts: list[str],
    policy: dict[str, Any],
    seed: int,
    max_candidates: int,
    n_views: int,
    random_draws: int,
    rng: np.random.Generator,
    cache_dir: Path | None = None,
    resume: bool = False,
    progress_label: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score, repair, and baseline each example with the frozen text-selected policy.

    The non-oracle scorer (``discover_clip_shortcut_regions``) receives image
    pixels, the prediction function, and class prompts only -- no labels, no
    harmful bbox, no shortcut type, no correctness. The frozen policy thresholds
    are applied verbatim.
    """

    class_names = CLIP_OVERLAY_CLASSES.copy()
    cert_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for index, ex in enumerate(examples, start=1):
        cache_key = f"example_{int(ex['example_id'])}_{_example_image_hash(ex)}"
        cert_path = cache_dir / f"{cache_key}_certificates.csv" if cache_dir is not None else None
        rank_path = cache_dir / f"{cache_key}_rankings.csv" if cache_dir is not None else None
        if resume and cert_path is not None and cert_path.exists() and rank_path is not None and rank_path.exists():
            cert_rows.extend(pd.read_csv(cert_path).to_dict("records"))
            ranking_rows.extend(pd.read_csv(rank_path).to_dict("records"))
            if progress_label:
                print(f"[{progress_label}] reused example {index}/{len(examples)}", flush=True)
            continue

        pil = _to_pil(ex["image"])
        predict_fn = _PredictionCache(model)
        # NON-ORACLE: pixels + predictions + prompts only.
        _, scores, original_probs = discover_clip_shortcut_regions(
            pil, predict_fn, prompts, seed=seed + int(ex["example_id"]), max_candidates=max_candidates
        )
        harmful_bbox = tuple(ex["harmful_bbox"]) if ex.get("harmful_bbox") else None

        ex_rank_rows: list[dict[str, Any]] = []
        for rank, score in enumerate(scores, start=1):
            row = score.to_dict()
            row.update(
                {
                    "example_id": ex["example_id"],
                    "split": ex["split"],
                    "regime": ex["regime"],
                    "rank": rank,
                    "harmful_iou": 0.0 if harmful_bbox is None else _iou(score.bbox, harmful_bbox),
                    "decoy_iou": max([_iou(score.bbox, b) for b in ex.get("decoy_bboxes", [])] or [0.0]),
                    "object_iou": _iou(score.bbox, ex["object_bbox"]),
                }
            )
            ex_rank_rows.append(row)

        top1 = scores[0] if scores else None
        top1_probs = predict_fn([neutralize_region(pil, top1.bbox)])[0] if top1 else original_probs
        top3_probs, top3_stable, top3_bbox, top3_id, top3_type = _consensus_repair(pil, scores[:3], predict_fn)

        # Generic non-text proposal baselines (no oracle, no class info).
        usable = [s for s in scores if s.proposal_type != "object_control"] or scores
        highest_contrast = max(usable, key=lambda s: s.edge_density, default=top1)
        largest = max(usable, key=lambda s: s.area_fraction, default=top1)
        object_controls = [s for s in scores if s.proposal_type == "object_control"]
        center_object = object_controls[0] if object_controls else top1
        highest_contrast_probs = predict_fn([neutralize_region(pil, highest_contrast.bbox)])[0] if highest_contrast else original_probs
        largest_probs = predict_fn([neutralize_region(pil, largest.bbox)])[0] if largest else original_probs
        center_object_probs = predict_fn([neutralize_region(pil, center_object.bbox)])[0] if center_object else original_probs
        aug_probs = _augment_prediction(model, ex, rng, n_views)
        oracle_probs = predict_fn([neutralize_region(pil, harmful_bbox)])[0] if harmful_bbox else original_probs

        # FROZEN policy decisions (no tuning).
        clean_probs, clean_abs, clean_bbox, clean_id, clean_type, clean_action = _policy_action(
            original_probs, top1_probs, top1, top3_probs, top3_stable, policy, allow_abstain=False
        )
        sel_probs, sel_abs, sel_bbox, sel_id, sel_type, sel_action = _policy_action(
            original_probs, top1_probs, top1, top3_probs, top3_stable, policy, allow_abstain=True
        )

        # Exploratory ablation: frozen policy WITHOUT the scorer's textness multiplier.
        # Not headline evidence -- reported only as an exploratory ablation.
        no_text_scores = sorted(scores, key=lambda s: s.score / max(s.textness_score, 1e-6), reverse=True)
        nt_top1 = no_text_scores[0] if no_text_scores else None
        nt_top1_probs = predict_fn([neutralize_region(pil, nt_top1.bbox)])[0] if nt_top1 else original_probs

        ex_cert_rows = [
            _row(ex, "original_clip_prediction", class_names, original_probs, original_probs, None, "none", "none", repair_action="keep_original"),
            _row(ex, "oracle_shortcut_neutralization", class_names, original_probs, oracle_probs, harmful_bbox, "oracle_harmful_bbox", "oracle upper bound", oracle=True),
            _row(ex, "frozen_cic_top1_repair", class_names, original_probs, top1_probs, top1.bbox if top1 else None, top1.candidate_id if top1 else "", top1.proposal_type if top1 else ""),
            _row(ex, "frozen_cic_top3_repair", class_names, original_probs, top3_probs, top3_bbox, top3_id, top3_type, abstained=not top3_stable, repair_action="repair" if top3_stable else "abstain"),
            _row(ex, "frozen_cic_clean_safe_repair", class_names, original_probs, clean_probs, clean_bbox, clean_id, clean_type, abstained=clean_abs, repair_action=clean_action),
            _row(ex, "frozen_cic_selective_repair_or_abstain", class_names, original_probs, sel_probs, sel_bbox, sel_id, sel_type, abstained=sel_abs, repair_action=sel_action),
            _row(ex, "highest_contrast_region_repair", class_names, original_probs, highest_contrast_probs, highest_contrast.bbox if highest_contrast else None, highest_contrast.candidate_id if highest_contrast else "", highest_contrast.proposal_type if highest_contrast else ""),
            _row(ex, "largest_region_repair", class_names, original_probs, largest_probs, largest.bbox if largest else None, largest.candidate_id if largest else "", largest.proposal_type if largest else ""),
            _row(ex, "center_object_region_repair", class_names, original_probs, center_object_probs, center_object.bbox if center_object else None, center_object.candidate_id if center_object else "", center_object.proposal_type if center_object else ""),
            _row(ex, "random_augmentation_consensus", class_names, original_probs, aug_probs, None, "random_augmentation", "random_augmentation"),
            _row(ex, "frozen_cic_top1_repair_no_textness_ablation", class_names, original_probs, nt_top1_probs, nt_top1.bbox if nt_top1 else None, nt_top1.candidate_id if nt_top1 else "", nt_top1.proposal_type if nt_top1 else ""),
        ]

        # Matched random region control: same area class as the proposed shortcut
        # region, random location avoiding the known badge boxes. Many draws.
        target_area = float(top1.area_fraction) if top1 else 0.04
        random_boxes = [(d, _random_patch_like(ex, target_area, rng)) for d in range(random_draws)]
        random_probs = predict_fn([neutralize_region(pil, box) for _, box in random_boxes])
        for (draw_id, box), probs in zip(random_boxes, random_probs):
            ex_cert_rows.append(_row(ex, "random_matched_region_repair", class_names, original_probs, probs, box, f"random_matched_{draw_id}", "random_matched_patch"))

        if cert_path is not None and rank_path is not None:
            pd.DataFrame(ex_cert_rows).to_csv(cert_path, index=False)
            pd.DataFrame(ex_rank_rows).to_csv(rank_path, index=False)
        cert_rows.extend(ex_cert_rows)
        ranking_rows.extend(ex_rank_rows)
        if progress_label:
            print(f"[{progress_label}] finished example {index}/{len(examples)} id={ex['example_id']} regime={ex['regime']}", flush=True)

    return pd.DataFrame(cert_rows), pd.DataFrame(ranking_rows)


# ---------------------------------------------------------------------------
# Failure-conditioned subset construction (Mode B)
# ---------------------------------------------------------------------------
def passes_failure_conditions(*, no_overlay_correct, aligned_correct, misleading_wrong, confidence_ok, oracle_restored) -> bool:
    return bool(no_overlay_correct and aligned_correct and misleading_wrong and confidence_ok and oracle_restored)


def build_failure_conditioned_set(
    cfg: dict[str, Any],
    predict_fn: _PredictionCache,
    *,
    size: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    pool_per_class = int(cfg.get("pool_per_class", 64))
    n_target = int(cfg.get("n_failure_target", 32))
    conf_threshold = float(cfg.get("original_confidence_threshold", 0.50))
    benchmark_seed = int(cfg.get("pool_benchmark_seed", 5151))
    n_classes = len(CLIP_OVERLAY_CLASSES)

    inclusion_rows: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    n_candidates = 0
    n_included = 0
    eid = benchmark_seed * 1_000_000

    for idx in range(pool_per_class):
        if n_included >= n_target:
            break
        specs = []
        images: list[Image.Image] = []
        for label in range(n_classes):
            mis = render_cross_shortcut_image(label, MIS_REGIME, idx, size=size, benchmark_seed=benchmark_seed)
            ali = render_cross_shortcut_image(label, ALIGNED_REGIME, idx, size=size, benchmark_seed=benchmark_seed)
            nov = render_cross_shortcut_image(label, NO_OVERLAY_REGIME, idx, size=size, benchmark_seed=benchmark_seed)
            specs.append((label, mis, ali, nov))
            images.extend([_to_pil(nov[0]), _to_pil(ali[0]), _to_pil(mis[0])])
        base_probs = predict_fn(images)
        oracle_images, oracle_index = [], []
        for spec_i, (_, mis, _, _) in enumerate(specs):
            bbox = mis[1].get("harmful_bbox") or []
            if bbox:
                oracle_images.append(neutralize_region(_to_pil(mis[0]), tuple(bbox)))
                oracle_index.append(spec_i)
        oracle_probs = predict_fn(oracle_images) if oracle_images else np.zeros((0, n_classes))
        oracle_lookup = {spec_i: oracle_probs[k] for k, spec_i in enumerate(oracle_index)}

        for spec_i, (label, mis, ali, nov) in enumerate(specs):
            n_candidates += 1
            nov_probs = base_probs[spec_i * 3 + 0]
            ali_probs = base_probs[spec_i * 3 + 1]
            mis_probs = base_probs[spec_i * 3 + 2]
            nov_correct = bool(int(nov_probs.argmax()) == label)
            ali_correct = bool(int(ali_probs.argmax()) == label)
            mis_wrong = bool(int(mis_probs.argmax()) != label)
            mis_conf = float(mis_probs.max())
            conf_ok = bool(mis_conf >= conf_threshold)
            oracle_correct = bool(spec_i in oracle_lookup and int(oracle_lookup[spec_i].argmax()) == label)
            passes = passes_failure_conditions(
                no_overlay_correct=nov_correct,
                aligned_correct=ali_correct,
                misleading_wrong=mis_wrong,
                confidence_ok=conf_ok,
                oracle_restored=oracle_correct,
            )
            included = bool(passes and n_included < n_target)
            reasons = []
            if not nov_correct:
                reasons.append("no_overlay_incorrect")
            if not ali_correct:
                reasons.append("aligned_incorrect")
            if not mis_wrong:
                reasons.append("misleading_not_wrong")
            if not conf_ok:
                reasons.append(f"confidence_below_{conf_threshold:g}")
            if not oracle_correct:
                reasons.append("oracle_did_not_restore")
            if included:
                reasons = []
            elif not reasons and n_included >= n_target:
                reasons.append("target_reached")
            inclusion_rows.append(
                {
                    "pool_index": idx,
                    "label": label,
                    "true_label": CLIP_OVERLAY_CLASSES[label],
                    "shortcut_type": SHORTCUT_TYPE,
                    "no_overlay_correct": nov_correct,
                    "aligned_correct": ali_correct,
                    "misleading_wrong": mis_wrong,
                    "misleading_confidence": mis_conf,
                    "confidence_threshold": conf_threshold,
                    "oracle_restored": oracle_correct,
                    "included": included,
                    "exclusion_reasons": ";".join(reasons),
                }
            )
            if not included:
                continue
            common = {"split": "failure_conditioned_test", "label": label, "true_label": CLIP_OVERLAY_CLASSES[label]}
            for regime, (arr, meta) in [(MIS_REGIME, mis), (ALIGNED_REGIME, ali), (NO_OVERLAY_REGIME, nov)]:
                failure_examples.append({"example_id": eid, "regime": regime, "image": arr, **common, **meta})
                eid += 1
            n_included += 1

    n_included = sum(1 for ex in failure_examples if ex["regime"] == MIS_REGIME)
    stats = {
        "n_candidates": int(n_candidates),
        "n_failure_examples": int(n_included),
        "inclusion_rate": float(n_included / n_candidates) if n_candidates else 0.0,
        "n_target": int(n_target),
        "original_confidence_threshold": conf_threshold,
        "pool_benchmark_seed": benchmark_seed,
        "pool_per_class": pool_per_class,
        "shortcut_type": SHORTCUT_TYPE,
        "target_reached": bool(n_included >= n_target),
    }
    return failure_examples, pd.DataFrame(inclusion_rows), stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _selective_accuracy(df: pd.DataFrame) -> tuple[float, float, float]:
    non_abs = ~df["abstained"].astype(bool)
    repaired = df.loc[non_abs, "repaired_correct"].astype(bool)
    coverage = float(non_abs.mean()) if len(df) else np.nan
    abst = float((~non_abs).mean()) if len(df) else np.nan
    acc = float(repaired.mean()) if bool(non_abs.sum()) else np.nan
    return acc, coverage, abst


def _repaired_acc(df: pd.DataFrame) -> tuple[float, int, int]:
    non_abs = ~df["abstained"].astype(bool)
    repaired = df.loc[non_abs, "repaired_correct"].astype(bool)
    return (float(repaired.mean()) if bool(non_abs.sum()) else np.nan, int(repaired.sum()), int(len(repaired)))


def _random_matched_stats(df: pd.DataFrame) -> dict[str, float]:
    if not len(df):
        return {}
    draw = df["selected_candidate_id"].astype(str).str.extract(r"random_matched_(\d+)")[0].fillna("0")
    acc, loc = [], []
    for _, sub in df.assign(_draw=draw).groupby("_draw"):
        non_abs = ~sub["abstained"].astype(bool)
        if bool(non_abs.sum()):
            acc.append(float(sub.loc[non_abs, "repaired_correct"].astype(bool).mean()))
        if "selected_harmful_iou" in sub:
            loc.append(float((sub["selected_harmful_iou"] >= 0.3).mean()))
    out: dict[str, float] = {}
    if acc:
        a = np.asarray(acc)
        out["random_matched_repair_mean"] = float(a.mean())
        out["random_matched_repair_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        out["random_matched_repair_ci95"] = float(1.96 * out["random_matched_repair_std"] / np.sqrt(len(a))) if len(a) > 1 else 0.0
    if loc:
        l = np.asarray(loc)
        out["random_matched_localization_iou_0_3_mean"] = float(l.mean())
        out["random_matched_localization_iou_0_3_std"] = float(l.std(ddof=1)) if len(l) > 1 else 0.0
        out["random_matched_localization_iou_0_3_ci95"] = float(1.96 * out["random_matched_localization_iou_0_3_std"] / np.sqrt(len(l))) if len(l) > 1 else 0.0
    return out


def _localization_metrics(rankings: pd.DataFrame) -> dict[str, float]:
    mis = rankings[rankings["regime"] == MIS_REGIME] if len(rankings) else pd.DataFrame()
    if not len(mis):
        return {}
    groups = [g for _, g in mis.groupby("example_id")]
    top1 = mis[mis["rank"] == 1]
    top1_03 = int((top1["harmful_iou"] >= 0.3).sum())
    top1_05 = int((top1["harmful_iou"] >= 0.5).sum())
    top3_03 = int(sum((g.nsmallest(3, "rank")["harmful_iou"] >= 0.3).any() for g in groups))
    top3_05 = int(sum((g.nsmallest(3, "rank")["harmful_iou"] >= 0.5).any() for g in groups))
    return {
        "harmful_top1_iou_0_3": float(top1_03 / max(1, len(top1))),
        "harmful_top1_iou_0_5": float(top1_05 / max(1, len(top1))),
        "harmful_top3_iou_0_3": float(top3_03 / max(1, len(groups))),
        "harmful_top3_iou_0_5": float(top3_05 / max(1, len(groups))),
        "harmful_top1_iou_0_3_ci95": _ci_text(top1_03, len(top1)),
        "median_harmful_rank": float(np.median([g.sort_values("harmful_iou", ascending=False)["rank"].iloc[0] for g in groups])),
    }


def natural_metrics(certs: pd.DataFrame, rankings: pd.DataFrame, status: ClipStatus) -> pd.DataFrame:
    rows = []
    loc = _localization_metrics(rankings)
    for method, df in certs.groupby("method", sort=False):
        row: dict[str, Any] = {
            "method": method,
            "backend": status.backend,
            "model_name": status.model_name,
            "pretrained_loaded": bool(status.pretrained),
            "shortcut_type": SHORTCUT_TYPE,
            "n_examples": int(len(df)),
        }
        for regime in CROSS_REGIMES:
            sub = df[df["regime"] == regime]
            key = regime.replace("cross_shortcut_", "")
            row[f"{key}_accuracy_before"] = float(sub["original_correct"].astype(bool).mean()) if len(sub) else np.nan
            acc, cov, _ = _selective_accuracy(sub) if len(sub) else (np.nan, np.nan, np.nan)
            row[f"{key}_accuracy_after"] = acc
        sel_acc, sel_cov, sel_abst = _selective_accuracy(df[df["regime"] == MIS_REGIME])
        row["misleading_selective_accuracy"] = sel_acc
        row["misleading_selective_coverage"] = sel_cov
        row["misleading_selective_abstention"] = sel_abst
        if method == "random_matched_region_repair":
            row.update(_random_matched_stats(df[df["regime"] == MIS_REGIME]))
        rows.append(row)
    out = pd.DataFrame(rows)
    for k, v in loc.items():
        out[k] = v
    return out


def failure_metrics(certs: pd.DataFrame, rankings: pd.DataFrame, status: ClipStatus, stats: dict[str, Any]) -> pd.DataFrame:
    rows = []
    loc = _localization_metrics(rankings)
    for method, df in certs.groupby("method", sort=False):
        mis = df[df["regime"] == MIS_REGIME]
        aligned = df[df["regime"] == ALIGNED_REGIME]
        nov = df[df["regime"] == NO_OVERLAY_REGIME]
        acc, succ, n_rep = _repaired_acc(mis) if len(mis) else (np.nan, 0, 0)
        sel_acc, sel_cov, sel_abst = _selective_accuracy(mis) if len(mis) else (np.nan, np.nan, np.nan)
        row: dict[str, Any] = {
            "method": method,
            "backend": status.backend,
            "model_name": status.model_name,
            "pretrained_loaded": bool(status.pretrained),
            "shortcut_type": SHORTCUT_TYPE,
            "n_failure_examples": int(len(mis)),
            "failure_subset_original_accuracy": float(mis["original_correct"].astype(bool).mean()) if len(mis) else np.nan,
            "failure_subset_repaired_accuracy": acc,
            "failure_subset_repaired_accuracy_ci95": _ci_text(succ, n_rep),
            "coverage": sel_cov,
            "abstention_rate": sel_abst,
            "selective_accuracy": sel_acc,
            "no_overlay_preservation_after": _selective_accuracy(nov)[0] if len(nov) else np.nan,
            "no_overlay_accuracy_before": float(nov["original_correct"].astype(bool).mean()) if len(nov) else np.nan,
            "aligned_preservation_after": _selective_accuracy(aligned)[0] if len(aligned) else np.nan,
            "aligned_accuracy_before": float(aligned["original_correct"].astype(bool).mean()) if len(aligned) else np.nan,
        }
        if method == "random_matched_region_repair" and len(mis):
            row.update(_random_matched_stats(mis))
        rows.append(row)
    out = pd.DataFrame(rows)
    for k, v in loc.items():
        out[k] = v
    out["n_candidates"] = stats["n_candidates"]
    out["inclusion_rate"] = stats["inclusion_rate"]
    return out


def repair_vs_localization(certs: pd.DataFrame, rankings: pd.DataFrame, label: str) -> pd.DataFrame:
    if certs.empty or rankings.empty:
        return pd.DataFrame()
    ids = certs.loc[(certs["method"] == "original_clip_prediction") & (certs["regime"] == MIS_REGIME), "example_id"].drop_duplicates()
    top1 = rankings[(rankings["regime"] == MIS_REGIME) & (rankings["rank"] == 1)].set_index("example_id")
    top3 = rankings[rankings["regime"] == MIS_REGIME].groupby("example_id")["harmful_iou"].apply(lambda s: bool((s.head(3) >= 0.3).any()))
    out = []
    for group, fn in [
        ("top1_iou_ge_0_3", lambda e: bool(e in top1.index and float(top1.loc[e, "harmful_iou"]) >= 0.3)),
        ("top1_iou_lt_0_3", lambda e: not bool(e in top1.index and float(top1.loc[e, "harmful_iou"]) >= 0.3)),
        ("top3_iou_ge_0_3", lambda e: bool(top3.get(e, False))),
        ("top3_iou_lt_0_3", lambda e: not bool(top3.get(e, False))),
    ]:
        gids = set(e for e in ids if fn(e))
        base = certs[certs["example_id"].isin(gids)]
        row = {"run_label": label, "group": group, "n_examples": len(gids)}
        for method, key in [
            ("original_clip_prediction", "original_accuracy"),
            ("frozen_cic_top1_repair", "cic_top1_repair_accuracy"),
            ("frozen_cic_top3_repair", "cic_top3_repair_accuracy"),
            ("random_matched_region_repair", "random_matched_repair_accuracy"),
        ]:
            sub = base[base["method"] == method]
            if method == "original_clip_prediction":
                row[key] = float(sub["original_correct"].astype(bool).mean()) if len(sub) else np.nan
            else:
                acc, _, _ = _repaired_acc(sub) if len(sub) else (np.nan, 0, 0)
                row[key] = acc
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Headline eligibility (PART 7)
# ---------------------------------------------------------------------------
def _val(d: dict[str, Any], k: str) -> float | None:
    v = d.get(k, np.nan)
    return float(v) if pd.notna(v) and np.isfinite(v) else None


def compute_headline_eligibility(
    natural: pd.DataFrame,
    failure: pd.DataFrame,
    fstats: dict[str, Any],
    status: ClipStatus,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    nat = natural.set_index("method").to_dict("index") if "method" in natural else {}
    fail = failure.set_index("method").to_dict("index") if "method" in failure else {}

    nat_mis_before = _val(nat.get("original_clip_prediction", {}), "misleading_accuracy_before")
    n_failure = int(fstats.get("n_failure_examples", 0))

    f_oracle = _val(fail.get("oracle_shortcut_neutralization", {}), "failure_subset_repaired_accuracy")
    f_top1 = _val(fail.get("frozen_cic_top1_repair", {}), "failure_subset_repaired_accuracy")
    f_top3 = _val(fail.get("frozen_cic_top3_repair", {}), "failure_subset_repaired_accuracy")
    f_clean = _val(fail.get("frozen_cic_clean_safe_repair", {}), "failure_subset_repaired_accuracy")
    rnd_row = fail.get("random_matched_region_repair", {})
    rnd_mean = _val(rnd_row, "random_matched_repair_mean")
    rnd_ci = _val(rnd_row, "random_matched_repair_ci95")
    best_cic = max([v for v in [f_top1, f_top3] if v is not None], default=None)

    gap = (best_cic - rnd_mean) if (best_cic is not None and rnd_mean is not None) else None
    cic_method = "frozen_cic_top1_repair" if (f_top1 is not None and (f_top3 is None or f_top1 >= f_top3)) else "frozen_cic_top3_repair"
    cic_n = int(fail.get(cic_method, {}).get("n_failure_examples", n_failure) or n_failure)
    cic_succ = int(round((best_cic or 0.0) * cic_n))
    cic_lo, _ = _wilson_ci(cic_succ, cic_n)
    non_overlapping = bool(best_cic is not None and rnd_mean is not None and cic_lo > (rnd_mean + (rnd_ci or 0.0)))
    beats_random = bool((gap is not None and gap >= 0.15) or non_overlapping)

    clean_min = float(cfg.get("clean_preservation_min", 0.85))
    nov_pres = _val(fail.get("frozen_cic_clean_safe_repair", {}), "no_overlay_preservation_after")
    aligned_pres = _val(fail.get("frozen_cic_clean_safe_repair", {}), "aligned_preservation_after")
    clean_preserved = bool(nov_pres is not None and aligned_pres is not None and nov_pres >= 0.90 and aligned_pres >= clean_min)

    n_or_natural_ok = bool(n_failure >= 30 or (nat_mis_before is not None and nat_mis_before <= 0.40))

    reasons: list[str] = []
    if not bool(status.pretrained):
        reasons.append("pretrained CLIP not loaded")
    if status.backend == "fake":
        reasons.append("fake backend")
    if not n_or_natural_ok:
        reasons.append(f"n_failure_examples {n_failure} < 30 and natural misleading accuracy not <= 0.40")
    if f_oracle is None or f_oracle < 0.85:
        reasons.append("oracle repair accuracy < 0.85")
    if best_cic is None or best_cic < 0.70:
        reasons.append("frozen CIC top-1/top-3 repair accuracy < 0.70")
    if not beats_random:
        reasons.append("frozen CIC does not beat matched random region repair by >= 0.15 or non-overlapping CI")
    if not clean_preserved:
        reasons.append("clean / no-overlay preservation not high")

    eligible = len(reasons) == 0
    key_numbers = {
        "cross_shortcut_headline_eligible": eligible,
        "cross_shortcut_headline_failed_reasons": reasons,
        "shortcut_type": SHORTCUT_TYPE,
        "evaluation_scope": "finite candidate non-text region proposals; not open-world shortcut discovery",
        "frozen_text_selected_policy_used": True,
        "no_cross_shortcut_tuning": True,
        "nonoracle_scorer_excludes_labels_bboxes_correctness": True,
        "pretrained_loaded": bool(status.pretrained),
        "fake_backend": status.backend == "fake",
        "backend": status.backend,
        "model_name": status.model_name,
        # Natural benchmark
        "natural_no_overlay_accuracy": _val(nat.get("original_clip_prediction", {}), "no_overlay_accuracy_before"),
        "natural_aligned_accuracy": _val(nat.get("original_clip_prediction", {}), "aligned_accuracy_before"),
        "natural_neutral_accuracy": _val(nat.get("original_clip_prediction", {}), "neutral_accuracy_before"),
        "natural_misleading_accuracy_before": nat_mis_before,
        "natural_oracle_repair_accuracy": _val(nat.get("oracle_shortcut_neutralization", {}), "misleading_accuracy_after"),
        "natural_cic_top1_repair_accuracy": _val(nat.get("frozen_cic_top1_repair", {}), "misleading_accuracy_after"),
        "natural_cic_top3_repair_accuracy": _val(nat.get("frozen_cic_top3_repair", {}), "misleading_accuracy_after"),
        "natural_cic_clean_safe_repair_accuracy": _val(nat.get("frozen_cic_clean_safe_repair", {}), "misleading_accuracy_after"),
        "natural_random_matched_repair_mean": _val(nat.get("random_matched_region_repair", {}), "random_matched_repair_mean"),
        "natural_random_matched_repair_95ci": _val(nat.get("random_matched_region_repair", {}), "random_matched_repair_ci95"),
        "natural_no_overlay_preservation_after": _val(nat.get("frozen_cic_clean_safe_repair", {}), "no_overlay_accuracy_after"),
        # Failure-conditioned
        "n_candidates": int(fstats.get("n_candidates", 0)),
        "n_failure_examples": n_failure,
        "inclusion_rate": float(fstats.get("inclusion_rate", 0.0)),
        "failure_subset_original_accuracy": _val(fail.get("original_clip_prediction", {}), "failure_subset_original_accuracy"),
        "oracle_repair_accuracy": f_oracle,
        "cic_top1_repair_accuracy": f_top1,
        "cic_top3_repair_accuracy": f_top3,
        "cic_clean_safe_repair_accuracy": f_clean,
        "cic_selective_accuracy": _val(fail.get("frozen_cic_selective_repair_or_abstain", {}), "selective_accuracy"),
        "cic_selective_coverage": _val(fail.get("frozen_cic_selective_repair_or_abstain", {}), "coverage"),
        "cic_selective_abstention": _val(fail.get("frozen_cic_selective_repair_or_abstain", {}), "abstention_rate"),
        "random_matched_repair_mean": rnd_mean,
        "random_matched_repair_std": _val(rnd_row, "random_matched_repair_std"),
        "random_matched_repair_95ci": rnd_ci,
        "cic_minus_random_gap": gap,
        "cic_beats_random": beats_random,
        "cic_non_overlapping_ci_vs_random": non_overlapping,
        "no_overlay_preservation_after": nov_pres,
        "aligned_preservation_after": aligned_pres,
        "clean_preservation_high": clean_preserved,
        # Localization
        "harmful_top1_iou_0_3": _val(failure.iloc[0].to_dict() if len(failure) else {}, "harmful_top1_iou_0_3"),
        "harmful_top1_iou_0_5": _val(failure.iloc[0].to_dict() if len(failure) else {}, "harmful_top1_iou_0_5"),
        "harmful_top3_iou_0_3": _val(failure.iloc[0].to_dict() if len(failure) else {}, "harmful_top3_iou_0_3"),
        "harmful_top3_iou_0_5": _val(failure.iloc[0].to_dict() if len(failure) else {}, "harmful_top3_iou_0_5"),
        "random_matched_localization_iou_0_3_mean": _val(rnd_row, "random_matched_localization_iou_0_3_mean"),
        "random_matched_localization_iou_0_3_95ci": _val(rnd_row, "random_matched_localization_iou_0_3_ci95"),
        # Exploratory ablation (not headline evidence)
        "exploratory_cic_top1_no_textness_accuracy": _val(fail.get("frozen_cic_top1_repair_no_textness_ablation", {}), "failure_subset_repaired_accuracy"),
    }
    return key_numbers, reasons


# ---------------------------------------------------------------------------
# Plot + summary + examples
# ---------------------------------------------------------------------------
def _plot(key_numbers: dict[str, Any], png: Path, pdf: Path) -> None:
    labels = ["original", "oracle", "CIC top1", "CIC top3", "random\nmatched"]
    vals = [
        key_numbers.get("failure_subset_original_accuracy") or 0.0,
        key_numbers.get("oracle_repair_accuracy") or 0.0,
        key_numbers.get("cic_top1_repair_accuracy") or 0.0,
        key_numbers.get("cic_top3_repair_accuracy") or 0.0,
        key_numbers.get("random_matched_repair_mean") or 0.0,
    ]
    colors = ["#888888", "#2c7", "#4c78a8", "#3a5e88", "#cc6677"]
    plt.figure(figsize=(8.2, 4.6))
    x = np.arange(len(labels))
    plt.bar(x, vals, color=colors)
    rci = key_numbers.get("random_matched_repair_95ci")
    if rci:
        plt.errorbar([4], [vals[4]], yerr=[rci], fmt="none", ecolor="black", capsize=4)
    plt.xticks(x, labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Failure-conditioned repair accuracy")
    title = "Cross-shortcut transfer (non-text watermark) — "
    title += "frozen text-selected CIC policy" if key_numbers.get("cross_shortcut_headline_eligible") else "frozen policy (NOT headline eligible)"
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NA"
    return f"{float(v):.3f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _write_summary(out_dir: Path, natural: pd.DataFrame, failure: pd.DataFrame, crosstab: pd.DataFrame, fstats: dict[str, Any], key_numbers: dict[str, Any], reasons: list[str], status: ClipStatus) -> None:
    k = key_numbers
    lines = [
        "# Cross-Shortcut Generalization Attempt",
        "",
        f"**{FROZEN_POLICY_LABEL}**",
        "",
        "This is a **finite-candidate** transfer test, **not open-world shortcut discovery** and not a claim of "
        "general robustness. The transfer test uses a finite candidate class of non-text region proposals and "
        "evaluates whether a CIC policy selected on text overlays transfers to one new shortcut family: a non-text "
        f"colored-symbol watermark ({SHORTCUT_TYPE}). The frozen repair/scoring policy (score threshold and "
        "consensus-stability threshold) is loaded verbatim from the hard multi-decoy text-overlay run and is NOT "
        "retuned, reselected, or reweighted on this benchmark. The non-oracle scorer receives image pixels, CLIP "
        "predictions, class prompts, and candidate proposals only -- never the true label, harmful shortcut bbox, "
        "shortcut type, or correctness. True labels and the harmful bbox are used only to define the held-out failure "
        "subset, the oracle upper bound, and localization metrics.",
        "",
        f"Backend: {status.backend}. Model: {status.model_name}. Pretrained loaded: `{status.pretrained}`. "
        f"Shortcut type: `{SHORTCUT_TYPE}` (non-text).",
        "",
        "## Mode A — Natural Cross-Shortcut Benchmark",
        "",
        f"- No-overlay accuracy: {_fmt(k['natural_no_overlay_accuracy'])}",
        f"- Aligned-shortcut accuracy: {_fmt(k['natural_aligned_accuracy'])}",
        f"- Neutral-shortcut accuracy: {_fmt(k['natural_neutral_accuracy'])}",
        f"- Original misleading accuracy: {_fmt(k['natural_misleading_accuracy_before'])}",
        f"- Oracle shortcut-neutralization repair (upper bound): {_fmt(k['natural_oracle_repair_accuracy'])}",
        f"- Frozen CIC top-1 / top-3 / clean-safe repair: {_fmt(k['natural_cic_top1_repair_accuracy'])} / {_fmt(k['natural_cic_top3_repair_accuracy'])} / {_fmt(k['natural_cic_clean_safe_repair_accuracy'])}",
        f"- Random matched region repair mean / 95% CI: {_fmt(k['natural_random_matched_repair_mean'])} / +/- {_fmt(k['natural_random_matched_repair_95ci'])}",
        f"- No-overlay preservation after clean-safe repair: {_fmt(k['natural_no_overlay_preservation_after'])}",
        "",
        "## Mode B — Failure-Conditioned Cross-Shortcut Evaluation",
        "",
        "**Failure-conditioned transfer evaluation, not natural benchmark accuracy.** Examples are admitted only when "
        "pretrained CLIP classifies the no-overlay and aligned images correctly, classifies the misleading non-text "
        "shortcut image incorrectly with confidence >= the configured threshold, and oracle shortcut neutralization "
        "restores the correct prediction. Original failure-subset accuracy is ~0 by construction.",
        "",
        f"- Candidates generated: {fstats['n_candidates']}",
        f"- Failure-conditioned examples included: {fstats['n_failure_examples']} (target {fstats['n_target']})",
        f"- Inclusion rate: {_fmt(fstats['inclusion_rate'])}",
        f"- Original failure-subset accuracy (~0 by construction): {_fmt(k['failure_subset_original_accuracy'])}",
        f"- Oracle shortcut-neutralization repair (upper bound): {_fmt(k['oracle_repair_accuracy'])}",
        f"- Frozen CIC top-1 repair: {_fmt(k['cic_top1_repair_accuracy'])}",
        f"- Frozen CIC top-3 repair: {_fmt(k['cic_top3_repair_accuracy'])}",
        f"- Frozen CIC clean-safe repair: {_fmt(k['cic_clean_safe_repair_accuracy'])}",
        f"- Frozen CIC selective accuracy / coverage / abstention: {_fmt(k['cic_selective_accuracy'])} / {_fmt(k['cic_selective_coverage'])} / {_fmt(k['cic_selective_abstention'])}",
        f"- Random matched region repair mean / std / 95% CI: {_fmt(k['random_matched_repair_mean'])} / {_fmt(k['random_matched_repair_std'])} / +/- {_fmt(k['random_matched_repair_95ci'])}",
        f"- CIC minus random matched gap: {_fmt(k['cic_minus_random_gap'])} (beats random: `{k['cic_beats_random']}`, non-overlapping CI: `{k['cic_non_overlapping_ci_vs_random']}`)",
        f"- No-overlay / aligned preservation after clean-safe repair: {_fmt(k['no_overlay_preservation_after'])} / {_fmt(k['aligned_preservation_after'])}",
        "",
        "### Exploratory ablation (not headline evidence)",
        "",
        f"- Frozen policy WITHOUT the scorer's textness term, CIC top-1 repair: {_fmt(k['exploratory_cic_top1_no_textness_accuracy'])}. "
        "This is reported only to check whether the inherited textness weighting helps or hurts transfer; it is not the main transfer result and the textness weight was not tuned.",
        "",
        "## Localization (failure subset)",
        "",
        f"- Frozen CIC top-1 harmful localization IoU >= 0.3 / 0.5: {_fmt(k['harmful_top1_iou_0_3'])} / {_fmt(k['harmful_top1_iou_0_5'])}",
        f"- Frozen CIC top-3 harmful localization IoU >= 0.3 / 0.5: {_fmt(k['harmful_top3_iou_0_3'])} / {_fmt(k['harmful_top3_iou_0_5'])}",
        f"- Random matched localization IoU >= 0.3 mean / 95% CI: {_fmt(k['random_matched_localization_iou_0_3_mean'])} / +/- {_fmt(k['random_matched_localization_iou_0_3_95ci'])}",
        "",
        "## Repair vs Localization Crosstab (failure subset)",
        "",
        _markdown_table(crosstab) if len(crosstab) else "Crosstab unavailable.",
        "",
        "## Headline Eligibility",
        "",
        f"- cross_shortcut_headline_eligible = `{k['cross_shortcut_headline_eligible']}`",
        f"- pretrained CLIP loaded: `{k['pretrained_loaded']}`; fake backend: `{k['fake_backend']}`",
        f"- frozen text-selected policy used: `{k['frozen_text_selected_policy_used']}`; no cross-shortcut tuning: `{k['no_cross_shortcut_tuning']}`",
        f"- non-oracle scorer excludes labels/bboxes/correctness: `{k['nonoracle_scorer_excludes_labels_bboxes_correctness']}`",
        ("- failed reasons: " + "; ".join(reasons)) if reasons else "- all headline-eligibility checks passed",
        "",
        (
            "**Conclusion: the frozen text-selected CIC policy transferred to a non-text shortcut type.** "
            "It repaired pretrained OpenCLIP failures on the non-text colored-symbol watermark better than matched random "
            "region repair while preserving clean performance. This remains a finite-candidate transfer test, not open-world discovery."
            if k["cross_shortcut_headline_eligible"]
            else "**Conclusion: the transfer attempt did not support cross-shortcut generalization.** "
            "The frozen text-selected policy did not clear all eligibility thresholds on the non-text shortcut. "
            "The main claim stays centered on text-region finite-candidate repair."
        ),
        "",
        "## Mode A Natural Metrics",
        "",
        _markdown_table(natural),
        "",
        "## Mode B Failure-Conditioned Metrics",
        "",
        _markdown_table(failure),
    ]
    (out_dir / "cross_shortcut_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_examples(out_dir: Path, inclusion_log: pd.DataFrame, failure_certs: pd.DataFrame) -> None:
    lines = [
        "# Cross-Shortcut Generalization Examples",
        "",
        f"Shortcut type: `{SHORTCUT_TYPE}` (non-text colored-symbol watermark).",
        "",
        "## Inclusion log (first 20 screened candidates)",
        "",
        _markdown_table(inclusion_log.head(20)) if len(inclusion_log) else "Unavailable.",
        "",
        "## Failure-subset repair certificates (first 20 rows)",
        "",
        _markdown_table(failure_certs.head(20)) if len(failure_certs) else "Unavailable.",
    ]
    (out_dir / "cross_shortcut_examples.md").write_text("\n".join(lines), encoding="utf-8")


def _write_caption(out_dir: Path, key_numbers: dict[str, Any]) -> None:
    eligible = key_numbers["cross_shortcut_headline_eligible"]
    caption = (
        "# Cross-Shortcut Generalization Figure Caption\n\n"
        f"Failure-conditioned repair accuracy on a non-text colored-symbol watermark shortcut ({SHORTCUT_TYPE}) using a "
        "CIC repair/scoring policy **selected on text-overlay shortcuts and frozen** (no retuning on this shortcut family). "
        "Bars: original (pre-repair, ~0 by construction), oracle shortcut neutralization (upper bound), frozen CIC top-1 and "
        "top-3 repair, and matched random region repair (error bar = 95% CI over random draws). "
        + (
            "The frozen text-selected policy transferred: it beat matched random region repair while preserving clean accuracy."
            if eligible
            else "The frozen policy did NOT clear the transfer eligibility thresholds; shown as a negative/limiting result."
        )
        + " This is a finite-candidate transfer test, not open-world shortcut discovery.\n"
    )
    (out_dir / "cross_shortcut_caption.md").write_text(caption, encoding="utf-8")


# ---------------------------------------------------------------------------
# Unavailable path
# ---------------------------------------------------------------------------
def _write_unavailable(out_dir: Path, status: ClipStatus, reason: str) -> dict[str, str]:
    for name in ["cross_shortcut_metrics.csv", "cross_shortcut_certificates.csv", "cross_shortcut_inclusion_log.csv", "cross_shortcut_repair_vs_localization.csv"]:
        pd.DataFrame([{"unavailable": True, "reason": reason, "backend": status.backend, "pretrained_loaded": bool(status.pretrained)}]).to_csv(out_dir / name, index=False)
    key_numbers = {
        "cross_shortcut_headline_eligible": False,
        "cross_shortcut_headline_failed_reasons": [reason],
        "shortcut_type": SHORTCUT_TYPE,
        "evaluation_scope": "finite candidate non-text region proposals; not open-world shortcut discovery",
        "pretrained_loaded": bool(status.pretrained),
        "fake_backend": status.backend == "fake",
        "frozen_text_selected_policy_used": False,
        "no_cross_shortcut_tuning": True,
        "reason": reason,
    }
    (out_dir / "cross_shortcut_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "cross_shortcut_summary.md").write_text(
        "# Cross-Shortcut Generalization Attempt\n\n"
        f"**{FROZEN_POLICY_LABEL}**\n\n"
        "This is a finite-candidate transfer test, not open-world shortcut discovery.\n\n"
        f"Pretrained CLIP unavailable or the frozen text-selected policy was missing ({reason}); no fake headline evidence was generated. "
        "The transfer attempt did not support cross-shortcut generalization under these conditions.\n",
        encoding="utf-8",
    )
    (out_dir / "cross_shortcut_examples.md").write_text("# Cross-Shortcut Examples\n\nUnavailable.\n", encoding="utf-8")
    _write_caption(out_dir, key_numbers)
    _plot(key_numbers, out_dir / "cross_shortcut_plot.png", out_dir / "cross_shortcut_plot.pdf")
    return {
        "metrics": str(out_dir / "cross_shortcut_metrics.csv"),
        "summary": str(out_dir / "cross_shortcut_summary.md"),
        "key_numbers": str(out_dir / "cross_shortcut_key_numbers.json"),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def make_natural_dataset(n_per_class: int, *, size: int, benchmark_seed: int, start_id: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    eid = start_id
    for regime in CROSS_REGIMES:
        for label in range(len(CLIP_OVERLAY_CLASSES)):
            for j in range(n_per_class):
                arr, meta = render_cross_shortcut_image(label, regime, j, size=size, benchmark_seed=benchmark_seed)
                examples.append({"example_id": eid, "split": "natural_test", "regime": regime, "image": arr, **meta})
                eid += 1
    return examples


def run(cfg: dict[str, Any]) -> dict[str, str]:
    total_start = time.perf_counter()
    timing: dict[str, float] = {}
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "cross_shortcut_generalization")
    size = int(cfg.get("data", {}).get("image_size", 200))

    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_clip_cross_shortcut", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for CLIP repair evidence")
        return _write_unavailable(out_dir, status, "fake backend is not allowed")
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, status, status.error_message or "pretrained CLIP did not load")

    # FROZEN policy from the text-overlay hard multi-decoy run. Never reselected here.
    frozen_dir = Path(cfg.get("frozen_policy_dir", "results/hard_multidecoy_clip_repair"))
    repair_policy = _frozen_policy(frozen_dir / "selected_repair_policy.json")
    if repair_policy is None:
        return _write_unavailable(out_dir, status, f"missing frozen text-selected repair policy at {frozen_dir / 'selected_repair_policy.json'}")
    print(f"[cross-shortcut] {FROZEN_POLICY_LABEL} threshold={repair_policy.get('score_threshold')} min_consensus={repair_policy.get('min_consensus_stability')}", flush=True)

    prompts = [PROMPT_TEMPLATE.format(label=name) for name in CLIP_OVERLAY_CLASSES]
    model = ClipZeroShotClassifier(status, CLIP_OVERLAY_CLASSES.copy(), prompts=prompts, device=device)

    max_candidates = int(cfg.get("max_candidates", 48))
    n_views = int(cfg.get("augmentation_views", 3))
    random_draws = int(cfg.get("random_draws", 25))
    resume = bool(cfg.get("resume", True))
    cache_root_dir = ensure_dir(out_dir / "example_eval_cache")

    # ---- Mode A: natural benchmark ----
    nat_n = int(cfg.get("data", {}).get("natural_n_per_class", 4))
    nat_seed = int(cfg.get("natural_benchmark_seed", 909))
    natural_examples = make_natural_dataset(nat_n, size=size, benchmark_seed=nat_seed, start_id=0)
    t0 = time.perf_counter()
    nat_certs, nat_rankings = evaluate_cross_shortcut(
        examples=natural_examples, model=model, prompts=prompts, policy=repair_policy, seed=seed,
        max_candidates=max_candidates, n_views=n_views, random_draws=random_draws, rng=rng,
        cache_dir=cache_root_dir / "natural", resume=resume, progress_label="natural",
    )
    timing["natural_eval_time_sec"] = time.perf_counter() - t0
    natural = natural_metrics(nat_certs, nat_rankings, status)

    # ---- Mode B: failure-conditioned ----
    screen_cache = _PredictionCache(model)
    t1 = time.perf_counter()
    failure_examples, inclusion_log, fstats = build_failure_conditioned_set(cfg, screen_cache, size=size)
    timing["screening_time_sec"] = time.perf_counter() - t1
    inclusion_log.to_csv(out_dir / "cross_shortcut_inclusion_log.csv", index=False)
    print(f"[cross-shortcut] failure screening: candidates={fstats['n_candidates']} included={fstats['n_failure_examples']} inclusion_rate={fstats['inclusion_rate']:.3f}", flush=True)

    if fstats["n_failure_examples"] > 0:
        t2 = time.perf_counter()
        fail_certs, fail_rankings = evaluate_cross_shortcut(
            examples=failure_examples, model=model, prompts=prompts, policy=repair_policy, seed=seed,
            max_candidates=max_candidates, n_views=n_views, random_draws=random_draws, rng=rng,
            cache_dir=cache_root_dir / "failure", resume=resume, progress_label="failure",
        )
        timing["failure_eval_time_sec"] = time.perf_counter() - t2
        failure = failure_metrics(fail_certs, fail_rankings, status, fstats)
        crosstab = repair_vs_localization(fail_certs, fail_rankings, "cross_shortcut_failure_conditioned")
    else:
        fail_certs, fail_rankings = pd.DataFrame(), pd.DataFrame()
        failure = pd.DataFrame([{"method": "no_failures_found", "n_failure_examples": 0, "shortcut_type": SHORTCUT_TYPE}])
        crosstab = pd.DataFrame()

    # ---- Persist certificates / metrics ----
    all_certs = pd.concat([nat_certs.assign(mode="natural"), fail_certs.assign(mode="failure_conditioned")], ignore_index=True) if len(fail_certs) else nat_certs.assign(mode="natural")
    all_certs.to_csv(out_dir / "cross_shortcut_certificates.csv", index=False)
    nat_rankings.assign(mode="natural").to_csv(out_dir / "cross_shortcut_candidate_rankings.csv", index=False)
    natural.assign(mode="natural").to_csv(out_dir / "cross_shortcut_metrics.csv", index=False)
    failure.assign(mode="failure_conditioned").to_csv(out_dir / "cross_shortcut_failure_metrics.csv", index=False)
    crosstab.to_csv(out_dir / "cross_shortcut_repair_vs_localization.csv", index=False)

    key_numbers, reasons = compute_headline_eligibility(natural, failure, fstats, status, cfg)
    timing["total_time_sec"] = time.perf_counter() - total_start
    key_numbers["timing_sec"] = {k: round(float(v), 3) for k, v in timing.items()}
    (out_dir / "cross_shortcut_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")

    _write_summary(out_dir, natural, failure, crosstab, fstats, key_numbers, reasons, status)
    _write_examples(out_dir, inclusion_log, fail_certs)
    _write_caption(out_dir, key_numbers)
    _plot(key_numbers, out_dir / "cross_shortcut_plot.png", out_dir / "cross_shortcut_plot.pdf")
    (out_dir / "cross_shortcut_generalization_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    print(f"[cross-shortcut] headline_eligible={key_numbers['cross_shortcut_headline_eligible']} reasons={reasons}", flush=True)
    print(f"[cross-shortcut] timing {timing}", flush=True)
    return {
        "metrics": str(out_dir / "cross_shortcut_metrics.csv"),
        "failure_metrics": str(out_dir / "cross_shortcut_failure_metrics.csv"),
        "certificates": str(out_dir / "cross_shortcut_certificates.csv"),
        "inclusion_log": str(out_dir / "cross_shortcut_inclusion_log.csv"),
        "repair_vs_localization": str(out_dir / "cross_shortcut_repair_vs_localization.csv"),
        "key_numbers": str(out_dir / "cross_shortcut_key_numbers.json"),
        "summary": str(out_dir / "cross_shortcut_summary.md"),
        "examples": str(out_dir / "cross_shortcut_examples.md"),
        "plot_png": str(out_dir / "cross_shortcut_plot.png"),
        "plot_pdf": str(out_dir / "cross_shortcut_plot.pdf"),
        "caption": str(out_dir / "cross_shortcut_caption.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cross_shortcut_generalization.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
