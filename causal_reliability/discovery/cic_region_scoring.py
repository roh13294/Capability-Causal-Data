from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from causal_reliability.discovery.region_proposals import RegionProposal


PredictionFn = Callable[[list[Image.Image]], np.ndarray]


@dataclass(frozen=True)
class RegionScore:
    candidate_id: str
    bbox: tuple[int, int, int, int]
    proposal_type: str
    score: float
    prediction_instability: float
    drop_in_original_top_class_probability: float
    prediction_flip_indicator: float
    kl_divergence_original_to_neutralized: float
    js_divergence: float
    distribution_shift_score: float
    alternative_prediction_confidence: float
    stable_alternative_bonus: float
    localization_specificity_penalty: float
    object_preservation_penalty: float
    confidence_collapse_penalty: float
    original_prediction_index: int
    neutralized_prediction_index: int
    original_confidence: float
    confidence_after_neutralization: float
    region_area_penalty: float
    specificity_score: float
    support_score: float
    textness_score: float
    consensus_stability: float
    object_preservation_score: float
    confidence_noncollapse_score: float
    clean_likelihood_penalty: float
    horizontalness_score: float
    center_overlap_score: float
    border_distance: float
    area_fraction: float
    edge_density: float
    center_distance: float
    width: int
    height: int
    aspect_ratio: float

    def to_dict(self) -> dict[str, object]:
        row = self.__dict__.copy()
        row["bbox"] = list(self.bbox)
        return row


def _local_background(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = bbox
    pad = max(4, int(0.035 * max(w, h)))
    sx0, sy0 = max(0, x0 - pad), max(0, y0 - pad)
    sx1, sy1 = min(w, x1 + pad), min(h, y1 + pad)
    surround = arr[sy0:sy1, sx0:sx1].copy()
    surround[max(0, y0 - sy0) : max(0, y1 - sy0), max(0, x0 - sx0) : max(0, x1 - sx0)] = np.nan
    flat = surround.reshape(-1, arr.shape[2])
    flat = flat[~np.isnan(flat).any(axis=1)]
    if len(flat) == 0:
        flat = arr.reshape(-1, arr.shape[2])
    return np.median(flat, axis=0)


def neutralize_region(image: Image.Image, bbox: tuple[int, int, int, int], strategy: str = "local_fill") -> Image.Image:
    pil = image.convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return pil.copy()
    if strategy == "blur":
        out = pil.copy()
        patch = out.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=max(2, int(0.04 * max(w, h)))))
        out.paste(patch, (x0, y0))
        return out
    fill = _local_background(arr, (x0, y0, x1, y1))
    if strategy == "inpaint_like":
        patch = np.zeros_like(arr[y0:y1, x0:x1])
        patch[:, :] = fill
        # Blend a little blurred context to avoid a hard synthetic edge.
        blurred = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=max(2, int(0.03 * max(w, h)))))).astype(np.float32) / 255.0
        arr[y0:y1, x0:x1] = 0.65 * patch + 0.35 * blurred[y0:y1, x0:x1]
    else:
        arr[y0:y1, x0:x1] = fill
    return Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))


def distribution_instability(original: np.ndarray, neutralized: np.ndarray) -> float:
    return float(0.5 * np.abs(original - neutralized).sum())


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-9
    pp = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    qq = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    pp = pp / pp.sum()
    qq = qq / qq.sum()
    return float(np.sum(pp * np.log(pp / qq)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    pp = np.asarray(p, dtype=np.float64)
    qq = np.asarray(q, dtype=np.float64)
    m = 0.5 * (pp + qq)
    return float(0.5 * _kl_divergence(pp, m) + 0.5 * _kl_divergence(qq, m))


def _weak_noop_variants(image: Image.Image) -> list[Image.Image]:
    return [
        image.filter(ImageFilter.GaussianBlur(radius=0.35)),
        image.filter(ImageFilter.UnsharpMask(radius=1, percent=40, threshold=3)),
    ]


def score_region_candidates(
    image: Image.Image,
    candidates: list[RegionProposal],
    predict_fn: PredictionFn,
) -> tuple[list[RegionScore], np.ndarray]:
    """Rank regions using only pixels, proposals, and model predictions.

    This function intentionally has no true-label, overlay-bbox, overlay-text, or
    test-correctness parameter. Evaluation metadata belongs downstream.
    """

    pil = image.convert("RGB")
    weak_variants = _weak_noop_variants(pil)
    variants: list[Image.Image] = [pil, *weak_variants]
    for cand in candidates:
        variants.extend(
            [
                neutralize_region(pil, cand.bbox, "local_fill"),
                neutralize_region(pil, cand.bbox, "blur"),
                neutralize_region(pil, cand.bbox, "inpaint_like"),
            ]
        )
    all_probs = np.asarray(predict_fn(variants), dtype=np.float64)
    original_probs = all_probs[0]
    original_pred = int(original_probs.argmax())
    original_conf = float(original_probs.max())
    weak_probs = all_probs[1 : 1 + len(weak_variants)]
    weak_preds = weak_probs.argmax(axis=1) if len(weak_probs) else np.asarray([], dtype=int)
    noop_stability = float((weak_preds == original_pred).mean()) if len(weak_preds) else 1.0
    noop_conf = float(weak_probs[:, original_pred].mean()) if len(weak_probs) else original_conf
    clean_likelihood = float(np.clip(0.5 * noop_stability + 0.5 * noop_conf, 0.0, 1.0))
    scored: list[RegionScore] = []
    for i, cand in enumerate(candidates):
        start = 1 + len(weak_variants) + i * 3
        probs = all_probs[start : start + 3]
        neutral = probs[0]
        neutral_pred = int(neutral.argmax())
        drops = np.maximum(0.0, original_probs[original_pred] - probs[:, original_pred])
        drop_original_top = float(drops.max())
        prediction_flip = float((probs.argmax(axis=1) != original_pred).any())
        js_values = np.asarray([_js_divergence(original_probs, row) for row in probs], dtype=float)
        kl_values = np.asarray([_kl_divergence(original_probs, row) for row in probs], dtype=float)
        distribution_shift = float(np.clip(js_values.max() / 0.35, 0.0, 1.0))
        instability = max(distribution_instability(original_probs, row) for row in probs)
        if neutral_pred != original_pred:
            instability = max(instability, 1.0)
        confidence_after = float(neutral.max())
        alternative_confidences = []
        for row in probs:
            alt = np.asarray(row, dtype=float).copy()
            alt[original_pred] = -np.inf
            alternative_confidences.append(float(np.max(alt)))
        alternative_confidence = float(max(alternative_confidences)) if alternative_confidences else 0.0
        preds = probs.argmax(axis=1)
        vals, counts = np.unique(preds, return_counts=True)
        best_pred = int(vals[int(counts.argmax())]) if len(vals) else neutral_pred
        stable_alternative = float((counts.max() / max(1, len(preds))) if len(counts) and best_pred != original_pred else 0.0)
        area_penalty = float(np.clip(1.0 - max(0.0, cand.area_fraction - 0.035) / 0.28, 0.08, 1.0))
        specificity = float(np.clip(1.0 - max(0.0, cand.area_fraction - 0.02) / 0.35, 0.05, 1.0))
        localization_specificity_penalty = float(1.0 - specificity)
        support = float(np.clip(1.0 - max(0.0, cand.area_fraction - 0.18) / 0.42, 0.05, 1.0))
        textness = float(np.clip(0.50 + 0.95 * cand.edge_density + 0.45 * cand.horizontalness_score + 0.25 * (1.0 - cand.border_distance), 0.25, 1.60))
        consensus = float((preds == neutral_pred).mean())
        object_preservation = float(np.clip(1.0 - 0.90 * cand.center_overlap_score - 0.25 * max(0.0, 0.55 - cand.center_distance), 0.05, 1.0))
        object_preservation_penalty = float(1.0 - object_preservation)
        confidence_noncollapse = float(np.clip((confidence_after - 0.18) / 0.55, 0.05, 1.0))
        confidence_collapse_penalty = float(1.0 - confidence_noncollapse)
        clean_penalty = float(np.clip(1.0 - 0.45 * clean_likelihood * (1.0 - min(instability, 1.0)), 0.35, 1.0))
        causal_effect = (
            drop_original_top
            + 0.55 * prediction_flip
            + 0.50 * distribution_shift
            + 0.30 * stable_alternative
            + 0.20 * alternative_confidence
        )
        score = causal_effect * specificity * support * consensus * textness * area_penalty * object_preservation * confidence_noncollapse * clean_penalty
        scored.append(
            RegionScore(
                candidate_id=cand.candidate_id,
                bbox=cand.bbox,
                proposal_type=cand.proposal_type,
                score=float(score),
                prediction_instability=float(instability),
                drop_in_original_top_class_probability=drop_original_top,
                prediction_flip_indicator=prediction_flip,
                kl_divergence_original_to_neutralized=float(kl_values.max()) if len(kl_values) else 0.0,
                js_divergence=float(js_values.max()) if len(js_values) else 0.0,
                distribution_shift_score=distribution_shift,
                alternative_prediction_confidence=alternative_confidence,
                stable_alternative_bonus=stable_alternative,
                localization_specificity_penalty=localization_specificity_penalty,
                object_preservation_penalty=object_preservation_penalty,
                confidence_collapse_penalty=confidence_collapse_penalty,
                original_prediction_index=original_pred,
                neutralized_prediction_index=neutral_pred,
                original_confidence=original_conf,
                confidence_after_neutralization=confidence_after,
                region_area_penalty=area_penalty,
                specificity_score=specificity,
                support_score=support,
                textness_score=textness,
                consensus_stability=consensus,
                object_preservation_score=object_preservation,
                confidence_noncollapse_score=confidence_noncollapse,
                clean_likelihood_penalty=clean_penalty,
                horizontalness_score=float(cand.horizontalness_score),
                center_overlap_score=float(cand.center_overlap_score),
                border_distance=float(cand.border_distance),
                area_fraction=float(cand.area_fraction),
                edge_density=float(cand.edge_density),
                center_distance=float(cand.center_distance),
                width=int(cand.width),
                height=int(cand.height),
                aspect_ratio=float(cand.aspect_ratio),
            )
        )
    return sorted(scored, key=lambda row: row.score, reverse=True), original_probs
