"""Lightweight CIC presentation/demo pipeline.

This module wires together the *real* finite-candidate CIC components so they can
be exercised interactively:

  * region proposals    -> causal_reliability.discovery.open_region_proposals
  * region scoring (CIC) -> causal_reliability.discovery.cic_region_scoring
  * region neutralization -> causal_reliability.discovery.cic_region_scoring
  * repair / abstention  -> causal_reliability.repair.repair_strategies

Only the *classifier* is swappable:

  * ``mode: real``  uses OpenCLIP zero-shot (downloads/loads real weights).
  * ``mode: mock``  uses a deterministic, model-free stub classifier. This is for
    demonstration and tests ONLY. Mock outputs are NOT scientific evidence.
  * ``mode: auto``  tries the real model and silently falls back to mock if the
    weights are unavailable; the result records which path actually ran.

Nothing here reads or writes benchmark result folders, support gates, or
``results/final_report/``. It is presentation scaffolding, not an experiment.

Scope note (also surfaced in the UI):
    CIC tests finite candidate interventions. It is not guaranteed open-world
    shortcut discovery or universal robustness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from causal_reliability.discovery.cic_region_scoring import (
    neutralize_region,
    score_region_candidates,
)
from causal_reliability.discovery.open_region_proposals import (
    generate_open_region_proposals,
)
from causal_reliability.repair.repair_strategies import (
    abstention_decision,
    counterfactual_consensus,
    intervention_instability,
    shortcut_neutralized_prediction,
)

SCOPE_NOTE = (
    "CIC tests finite candidate interventions. It is not guaranteed open-world "
    "shortcut discovery or universal robustness."
)

DEMO_DISCLAIMER = (
    "Demonstration tool for STS judging and small usability feedback only. "
    "Numbers shown here are illustrative of the workflow, not benchmark results. "
    "In mock mode the classifier is a deterministic stub and outputs are NOT "
    "scientific evidence."
)

# The three region neutralization strategies the real CIC scorer uses.
NEUTRALIZE_STRATEGIES = ("local_fill", "blur", "inpaint_like")

DEFAULT_CLASS_NAMES = [
    "tabby cat",
    "golden retriever dog",
    "ceramic coffee mug",
    "printed street sign",
    "open paperback book",
    "laptop computer",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DemoConfig:
    mode: str = "mock"  # "mock" | "real" | "auto"
    class_names: list[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    max_candidates: int = 32
    seed: int = 0
    top_k: int = 5
    device: str = "cpu"
    allow_download: bool = False
    preferred_backend: str = "open_clip"
    high_confidence_threshold: float = 0.8
    low_stability_threshold: float = 0.5
    export_dir: str = "results/demo_validation/exports"
    sample_images_dir: str = "demo/sample_images"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DemoConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    mode_requested: str
    mode_used: str  # "real" or "mock"
    backend: str
    class_names: list[str]
    original_top_k: list[dict[str, float]]
    repaired_top_k: list[dict[str, float]]
    selected_region: dict[str, Any]
    cic_selected_score: float
    prediction_changed: bool
    repair_strategy: str
    selected_intervention: str
    repaired_confidence: float
    stability_score: float
    reliability_action: str  # "repair" | "abstain"
    reliability_reason: str
    n_candidates: int
    scope_note: str = SCOPE_NOTE
    disclaimer: str = DEMO_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
def _pil_batch_to_tensor(images: list[Image.Image], size: int = 224):
    import torch

    arrays = []
    for img in images:
        arr = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        arrays.append(np.transpose(arr, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrays))


def build_real_predict_fn(config: DemoConfig):
    """Return ``(predict_fn, backend)`` backed by real OpenCLIP, or ``(None, msg)``."""
    from causal_reliability.real_models.clip_zero_shot import (
        ClipZeroShotClassifier,
        check_clip_available,
    )

    status = check_clip_available(
        device=config.device,
        allow_download=config.allow_download,
        preferred_backend=config.preferred_backend,
    )
    if not status.available:
        return None, status.error_message

    classifier = ClipZeroShotClassifier(status, list(config.class_names), device=config.device)

    def predict_fn(images: list[Image.Image]) -> np.ndarray:
        batch = _pil_batch_to_tensor(images)
        out = classifier.predict(batch)
        return out["probabilities"].numpy()

    return predict_fn, status.backend


def _edge_energy(image: Image.Image) -> float:
    """Model-free proxy for high-frequency (often text) content in [0, 1]."""
    from PIL import ImageFilter

    edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    return float(edges.mean())


def build_mock_predict_fn(config: DemoConfig) -> Callable[[list[Image.Image]], np.ndarray]:
    """Deterministic, model-free stub classifier.

    DEMO/TEST ONLY. It assigns probability mass to text-like classes in
    proportion to high-frequency edge content and to object classes from mean
    colour. Removing a text region therefore lowers the text-class score and can
    flip the prediction -- mirroring the *shape* of a CIC shortcut repair without
    being a real model. Outputs are not scientific evidence.
    """
    names = list(config.class_names)
    n = len(names)
    text_matches = [i for i, x in enumerate(names) if any(t in x.lower() for t in ("sign", "book", "text"))]
    primary_text = text_matches[0] if text_matches else min(3, n - 1)
    # Energy is tiny in absolute terms (~0.005-0.04); centre and scale it so a
    # text-heavy region drives the shortcut class to high confidence and its
    # removal collapses that class below the colour-driven object classes.
    e_mid, e_scale = 0.023, 320.0

    def predict_fn(images: list[Image.Image]) -> np.ndarray:
        rows = []
        for img in images:
            rgb = img.convert("RGB").resize((96, 96))
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            mean_rgb = arr.reshape(-1, 3).mean(axis=0)
            energy = _edge_energy(rgb)
            logits = np.zeros(n, dtype=np.float64)
            # Object-class logits from colour channels (deterministic, stable).
            dominant = int(np.argmax(mean_rgb))
            for i in range(n):
                logits[i] = 1.6 * mean_rgb[i % 3] + 0.9 * ((i % 3) == dominant)
            # Shortcut (text) class driven by edge energy -> the "shortcut" signal.
            text_drive = e_scale * (energy - e_mid)
            logits[primary_text] += text_drive
            for ti in text_matches[1:]:
                logits[ti] += 0.4 * text_drive
            logits -= logits.max()
            exp = np.exp(logits)
            rows.append(exp / exp.sum())
        return np.asarray(rows, dtype=np.float64)

    return predict_fn


def resolve_predict_fn(config: DemoConfig) -> tuple[Callable[[list[Image.Image]], np.ndarray], str, str]:
    """Return ``(predict_fn, mode_used, backend)`` honouring ``config.mode``."""
    if config.mode == "mock":
        return build_mock_predict_fn(config), "mock", "deterministic_stub"
    if config.mode in ("real", "auto"):
        predict_fn, backend = build_real_predict_fn(config)
        if predict_fn is not None:
            return predict_fn, "real", backend
        if config.mode == "real":
            raise RuntimeError(
                f"Real CLIP model unavailable: {backend}. "
                "Install open_clip_torch and allow weight download, or use mode: mock."
            )
        # auto -> fall back to mock
        return build_mock_predict_fn(config), "mock", "deterministic_stub"
    raise ValueError(f"unknown mode: {config.mode!r} (expected mock|real|auto)")


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def _top_k(probs: np.ndarray, names: list[str], k: int) -> list[dict[str, float]]:
    order = np.argsort(probs)[::-1][:k]
    return [{"label": names[i], "confidence": float(probs[i])} for i in order]


def run_pipeline(
    image: Image.Image,
    config: DemoConfig,
    predict_fn: Callable[[list[Image.Image]], np.ndarray] | None = None,
) -> PipelineResult:
    """Run the finite-candidate CIC demo flow on a single image.

    Region proposal generation, region scoring, neutralization and repair are the
    *real* project components. Only the classifier may be a deterministic stub
    (mock mode). This is a demonstration, not an experiment: it does not touch
    benchmark metrics, support gates or results/final_report/.
    """
    image = image.convert("RGB")
    names = list(config.class_names)
    k = min(config.top_k, len(names))

    mode_used = "mock"
    backend = "deterministic_stub"
    if predict_fn is None:
        predict_fn, mode_used, backend = resolve_predict_fn(config)
    elif config.mode == "real":
        mode_used, backend = "real", "provided"

    # 1) finite candidate proposals (model-free, real proposal code)
    proposals = generate_open_region_proposals(
        image, seed=config.seed, max_candidates=config.max_candidates
    )

    # 2) original prediction
    original_probs = np.asarray(predict_fn([image]), dtype=np.float64)[0]
    original_top_k = _top_k(original_probs, names, k)

    if not proposals:
        # Degenerate image with no candidate regions: nothing to repair.
        return PipelineResult(
            mode_requested=config.mode,
            mode_used=mode_used,
            backend=backend,
            class_names=names,
            original_top_k=original_top_k,
            repaired_top_k=original_top_k,
            selected_region={},
            cic_selected_score=0.0,
            prediction_changed=False,
            repair_strategy="none",
            selected_intervention="none",
            repaired_confidence=float(original_probs.max()),
            stability_score=1.0,
            reliability_action="repair",
            reliability_reason="no candidate regions; original prediction kept",
            n_candidates=0,
        )

    # 3) score proposals with CIC (real scorer; uses original + neutralized preds)
    scored, original_probs_scored = score_region_candidates(image, proposals, predict_fn)
    original_probs = np.asarray(original_probs_scored, dtype=np.float64)
    original_top_k = _top_k(original_probs, names, k)
    selected = scored[0]

    # 4) build counterfactuals for the selected region across the 3 strategies
    cf_images = [neutralize_region(image, selected.bbox, s) for s in NEUTRALIZE_STRATEGIES]
    cf_probs = np.asarray(predict_fn(cf_images), dtype=np.float64)

    # 5) repair via real repair strategy + abstention policy
    repair = shortcut_neutralized_prediction(
        original_probs, cf_probs, intervention_names=list(NEUTRALIZE_STRATEGIES)
    )
    consensus = counterfactual_consensus(cf_probs, intervention_names=list(NEUTRALIZE_STRATEGIES))
    decision = abstention_decision(
        original_confidence=float(original_probs.max()),
        stability_score=consensus.consensus_stability,
        consensus=consensus,
        high_confidence_threshold=config.high_confidence_threshold,
        low_stability_threshold=config.low_stability_threshold,
    )

    # Repaired top-5 = the counterfactual the repair strategy selected as most
    # informative (highest intervention instability).
    instability = intervention_instability(
        np.repeat(original_probs.reshape(1, -1), len(cf_probs), axis=0), cf_probs
    )
    repaired_probs = cf_probs[int(np.argmax(instability))]
    repaired_top_k = _top_k(repaired_probs, names, k)

    prediction_changed = int(repaired_probs.argmax()) != int(original_probs.argmax())

    selected_region = {
        "bbox": list(selected.bbox),
        "proposal_type": selected.proposal_type,
        "candidate_id": selected.candidate_id,
        "area_fraction": float(selected.area_fraction),
        "prediction_instability": float(selected.prediction_instability),
        "js_divergence": float(selected.js_divergence),
    }

    return PipelineResult(
        mode_requested=config.mode,
        mode_used=mode_used,
        backend=backend,
        class_names=names,
        original_top_k=original_top_k,
        repaired_top_k=repaired_top_k,
        selected_region=selected_region,
        cic_selected_score=float(selected.score),
        prediction_changed=bool(prediction_changed),
        repair_strategy=repair.repair_strategy,
        selected_intervention=repair.selected_intervention,
        repaired_confidence=float(repair.repaired_confidence),
        stability_score=float(consensus.consensus_stability),
        reliability_action=decision.repair_action,
        reliability_reason=(
            "high confidence + low stability -> abstain"
            if decision.repair_action == "abstain"
            else "stable consensus -> repaired prediction accepted"
        ),
        n_candidates=len(proposals),
    )


# --------------------------------------------------------------------------- #
# Visualization + export
# --------------------------------------------------------------------------- #
def render_region_overlay(image: Image.Image, result: PipelineResult) -> Image.Image:
    """Return a copy of ``image`` with the selected CIC region outlined."""
    from PIL import ImageDraw

    out = image.convert("RGB").copy()
    bbox = result.selected_region.get("bbox")
    if bbox and len(bbox) == 4:
        draw = ImageDraw.Draw(out)
        draw.rectangle([int(v) for v in bbox], outline=(220, 30, 30), width=max(2, out.width // 120))
    return out


def export_report(
    image: Image.Image,
    result: PipelineResult,
    export_dir: str | Path,
    name: str = "cic_demo_report",
) -> dict[str, str]:
    """Write a small JSON + PNG report. Returns the written paths.

    Writes only inside ``export_dir`` (a demo output folder). Never touches
    benchmark result folders or results/final_report/.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{name}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)

    png_path = out_dir / f"{name}.png"
    _render_report_png(image, result, png_path)

    return {"json": str(json_path), "png": str(png_path)}


def _render_report_png(image: Image.Image, result: PipelineResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay = render_region_overlay(image, result)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"CIC region ({result.selected_region.get('proposal_type', 'n/a')})")
    axes[1].axis("off")

    axes[2].axis("off")
    lines = [
        f"mode: {result.mode_used} ({result.backend})",
        "",
        "Original top-1: "
        + (f"{result.original_top_k[0]['label']} "
           f"({result.original_top_k[0]['confidence']:.2f})" if result.original_top_k else "n/a"),
        "Repaired top-1: "
        + (f"{result.repaired_top_k[0]['label']} "
           f"({result.repaired_top_k[0]['confidence']:.2f})" if result.repaired_top_k else "n/a"),
        f"prediction changed: {result.prediction_changed}",
        f"stability: {result.stability_score:.2f}",
        f"reliability: {result.reliability_action}",
        "",
        "SCOPE: " + result.scope_note,
    ]
    axes[2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, wrap=True)

    fig.suptitle("CIC finite-candidate reliability demo (illustrative, not benchmark evidence)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
