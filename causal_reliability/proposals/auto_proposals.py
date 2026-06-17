from __future__ import annotations

"""Automated finite-candidate proposal generators for CIC.

Each generator turns an image into a finite set of candidate bounding boxes
**from pixels alone** (no true label, shortcut identity, OCR text content,
correctness, or benchmark condition). The boxes are then converted to
``RegionProposal`` objects and scored by the existing, model-agnostic
``cic_region_scoring.score_region_candidates`` logic. CIC therefore still scores
a *finite* candidate set; the only difference from the manually-designed
candidate sets used elsewhere is that the candidate set is generated
automatically.

Generators
----------
* ``grid_boxes``            - simple multi-scale grid / open-region boxes (always available).
* ``edge_component_boxes``  - classical connected-component boxes using installed
                              dependencies only (numpy / PIL / scipy).
* ``saliency_boxes``        - image-gradient / edge saliency boxes (numpy / scipy).
* ``sam_boxes``             - optional SAM / SAM2 adapter; skips cleanly if SAM is
                              not installed (``available=False`` + ``skip_reason``).
* ``dino_boxes``            - optional GroundingDINO / DINO detector adapter; skips
                              cleanly if not installed.

Dependency policy
-----------------
* The classical generators (``grid``/``edge_component``/``saliency``) require only
  numpy, PIL, and scipy and are always available.
* Optional adapters (``sam``/``dino``) NEVER import a heavy optional dependency at
  module import time, never trigger a model download unless ``allow_download=True``
  is explicitly passed, and return ``ProposalSet(available=False, skip_reason=...)``
  when the dependency is missing. Tests therefore pass without SAM / DINO.

This module is **not** open-world shortcut discovery: it produces a finite
candidate set. No universal-robustness or deployment claim is implied here.
"""

import hashlib
import importlib.util
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from causal_reliability.discovery.region_proposals import (
    BBox,
    RegionProposal,
    _clip_bbox,
    _dedupe,
    _gray_edges,
    proposal_from_bbox,
    random_patch_proposals,
    text_box_component_proposals,
    textness_proposals,
)
from causal_reliability.discovery.open_region_proposals import grid_patch_proposals


# Proposal-type labels used per family. These feed ``RegionProposal.proposal_type``
# and are how downstream experiments distinguish the generating family.
GRID_TYPE = "auto_grid_box"
EDGE_TYPE = "auto_edge_component_box"
SALIENCY_TYPE = "auto_saliency_box"
SAM_TYPE = "auto_sam_box"
DINO_TYPE = "auto_dino_box"
RANDOM_TYPE = "random_patch_control"

CLASSICAL_FAMILIES = ("grid_boxes", "edge_component_boxes", "saliency_boxes")
OPTIONAL_FAMILIES = ("sam_boxes", "dino_boxes")
ALL_FAMILIES = CLASSICAL_FAMILIES + OPTIONAL_FAMILIES

# Real-SAM defaults. The checkpoint is NOT committed (see .gitignore: models/sam/);
# it must be downloaded manually and placed here. Python code never downloads it.
DEFAULT_SAM_CHECKPOINT = "models/sam/sam_vit_b_01ec64.pth"
DEFAULT_SAM_MODEL_TYPE = "vit_b"


@dataclass
class ProposalSet:
    """A finite candidate box set from one generator family.

    ``available`` is ``False`` (with a human-readable ``skip_reason``) when an
    optional dependency is missing; the classical families are always available.
    """

    name: str
    proposal_type: str
    boxes: list[BBox] = field(default_factory=list)
    available: bool = True
    skip_reason: str = ""
    # Caching / timing telemetry (used by the SAM fast path; defaults keep the
    # classical generators' behaviour unchanged).
    cache_hit: bool | None = None
    gen_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["boxes"] = [list(b) for b in self.boxes]
        row["n_boxes"] = len(self.boxes)
        return row


@dataclass
class SamConfig:
    """Configuration for the real Segment-Anything proposal generator.

    Every knob is exposed so the COCO sweep can tune SAM without editing code. The
    defaults match the finalized finite-candidate budget (``max_proposals=48``) and
    use ``vit_b`` with the committed-but-gitignored checkpoint path.
    """

    checkpoint_path: str = DEFAULT_SAM_CHECKPOINT
    model_type: str = DEFAULT_SAM_MODEL_TYPE
    points_per_side: int = 16
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.90
    max_proposals: int = 48
    min_area_frac: float = 0.002
    max_area_frac: float = 0.80
    min_side: int = 8
    device: str = "auto"
    dedupe_iou: float = 0.7
    # Fast-path knobs: ``crop_n_layers=0`` disables SAM's expensive multi-crop pass;
    # ``max_side`` downscales the longest image side before SAM (0 = no resize).
    crop_n_layers: int = 0
    max_side: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray((np.asarray(image).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def _module_installed(name: str) -> bool:
    """True iff a module can be located WITHOUT importing it (no side effects)."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


# --------------------------------------------------------------------------- #
# Classical generators (always available)
# --------------------------------------------------------------------------- #
def grid_boxes(
    image: Image.Image | np.ndarray,
    *,
    scales: list[float] | None = None,
    overlap: float = 0.5,
    max_boxes: int = 48,
    **_: object,
) -> ProposalSet:
    """Simple multi-scale grid / open-region boxes tiling the whole image."""

    pil = _as_pil(image)
    edges = _gray_edges(pil)
    props = grid_patch_proposals(pil, edges, scales, overlap=overlap)
    boxes = [p.bbox for p in props][: max(1, int(max_boxes))]
    return ProposalSet(name="grid_boxes", proposal_type=GRID_TYPE, boxes=boxes, available=True)


def edge_component_boxes(
    image: Image.Image | np.ndarray,
    *,
    max_boxes: int = 48,
    **_: object,
) -> ProposalSet:
    """Lightweight classical connected-component proposals (numpy / PIL only).

    Reuses the high-contrast and high-frequency connected-component detectors that
    already ship in ``region_proposals`` — these are deterministic classical image
    operations that do not require OpenCV or any model.
    """

    pil = _as_pil(image)
    edges = _gray_edges(pil)
    props = list(text_box_component_proposals(pil, edges, max_components=max(8, int(max_boxes))))
    props.extend(textness_proposals(pil, edges, max_components=max(8, int(max_boxes))))
    seen: set[BBox] = set()
    boxes: list[BBox] = []
    for p in props:
        if p.bbox in seen:
            continue
        seen.add(p.bbox)
        boxes.append(p.bbox)
    return ProposalSet(
        name="edge_component_boxes",
        proposal_type=EDGE_TYPE,
        boxes=boxes[: max(1, int(max_boxes))],
        available=True,
    )


def _connected_components_from_mask(mask: np.ndarray) -> list[tuple[int, BBox]]:
    """Return ``(size, bbox)`` for each connected component of a boolean mask.

    Uses ``scipy.ndimage.label`` when available (it always is here), with a pure
    numpy flood-fill fallback so the generator never hard-depends on scipy.
    """

    h, w = mask.shape
    try:
        from scipy import ndimage

        labels, n = ndimage.label(mask)
        out: list[tuple[int, BBox]] = []
        if n == 0:
            return out
        objs = ndimage.find_objects(labels)
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            ys, xs = sl
            size = int((labels[sl] == i).sum())
            out.append((size, (int(xs.start), int(ys.start), int(xs.stop), int(ys.stop))))
        return out
    except Exception:
        # Pure-numpy fallback (iterative flood fill).
        visited = np.zeros_like(mask, dtype=bool)
        out = []
        for y0 in range(h):
            for x0 in range(w):
                if visited[y0, x0] or not mask[y0, x0]:
                    continue
                stack = [(x0, y0)]
                visited[y0, x0] = True
                xs_l: list[int] = []
                ys_l: list[int] = []
                while stack:
                    sx, sy = stack.pop()
                    xs_l.append(sx)
                    ys_l.append(sy)
                    for nx in (sx - 1, sx, sx + 1):
                        for ny in (sy - 1, sy, sy + 1):
                            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and mask[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((nx, ny))
                out.append((len(xs_l), (min(xs_l), min(ys_l), max(xs_l) + 1, max(ys_l) + 1)))
        return out


def saliency_boxes(
    image: Image.Image | np.ndarray,
    *,
    max_boxes: int = 24,
    quantile: float = 0.85,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.45,
    **_: object,
) -> ProposalSet:
    """Image-gradient / edge saliency proposals.

    Computes a smoothed gradient-magnitude saliency map, thresholds it at a high
    quantile, and returns bounding boxes of the most-salient connected regions.
    Uses only numpy / PIL / scipy that are already installed dependencies.
    """

    pil = _as_pil(image)
    edges = _gray_edges(pil)
    h, w = edges.shape
    # Smooth the saliency map so boxes track coherent salient blobs, not pixels.
    try:
        from scipy.ndimage import uniform_filter

        sal = uniform_filter(edges.astype(np.float32), size=max(3, int(0.04 * max(w, h))))
    except Exception:
        k = max(3, int(0.04 * max(w, h)))
        kernel = np.ones((k, k), dtype=np.float32) / float(k * k)
        sal = _box_blur(edges.astype(np.float32), kernel)
    thr = float(np.quantile(sal, float(quantile)))
    mask = sal > thr
    comps = _connected_components_from_mask(mask)
    total = float(max(1, w * h))
    scored: list[tuple[float, BBox]] = []
    for size, (x0, y0, x1, y1) in comps:
        bbox = _clip_bbox((x0, y0, x1, y1), w, h)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        area = float(bw * bh) / total
        if area < float(min_area_frac) or area > float(max_area_frac):
            continue
        mean_sal = float(sal[bbox[1] : bbox[3], bbox[0] : bbox[2]].mean()) if bw and bh else 0.0
        scored.append((mean_sal * (0.5 + 0.5 * min(1.0, size / (0.02 * total))), bbox))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    boxes = [b for _, b in scored][: max(1, int(max_boxes))]
    return ProposalSet(name="saliency_boxes", proposal_type=SALIENCY_TYPE, boxes=boxes, available=True)


def _box_blur(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Tiny separable-free convolution fallback (only used if scipy is missing).
    from numpy.lib.stride_tricks import sliding_window_view

    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = sliding_window_view(padded, (k, k))
    return (windows * kernel).sum(axis=(-1, -2))


# --------------------------------------------------------------------------- #
# Optional adapters (skip cleanly when the dependency is absent)
# --------------------------------------------------------------------------- #
# A real SAM model is loaded at most once per (checkpoint, model_type, settings,
# device) and cached so per-image generation does not reload 375MB of weights.
_SAM_CACHE: dict[tuple, object] = {}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to cuda > mps > cpu; pass an explicit device through."""

    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _build_sam_mask_generator(cfg: SamConfig):
    """Build (and cache) a ``SamAutomaticMaskGenerator`` for ``cfg``.

    Imports ``segment_anything`` lazily so module import never pulls in SAM, and
    never downloads weights (the checkpoint must already exist on disk)."""

    device = _resolve_device(cfg.device)
    key = (
        cfg.checkpoint_path,
        cfg.model_type,
        int(cfg.points_per_side),
        float(cfg.pred_iou_thresh),
        float(cfg.stability_score_thresh),
        int(cfg.crop_n_layers),
        device,
    )
    cached = _SAM_CACHE.get(key)
    if cached is not None:
        return cached, device

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint_path)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=int(cfg.points_per_side),
        pred_iou_thresh=float(cfg.pred_iou_thresh),
        stability_score_thresh=float(cfg.stability_score_thresh),
        crop_n_layers=int(cfg.crop_n_layers),
    )
    _SAM_CACHE[key] = generator
    return generator, device


def _run_sam(pil: Image.Image, cfg: SamConfig) -> list[dict]:
    """Generate SAM masks for one image and return raw XYXY candidate dicts.

    Optionally downscales the longest side to ``cfg.max_side`` before SAM (a large
    speedup), then rescales every box back to the *original* image coordinates so
    downstream area/side filtering uses the true image size. Falls back to CPU on
    device errors. Each returned dict has ``bbox`` (XYXY, original coords),
    ``predicted_iou`` and ``stability_score``.
    """

    rgb = pil.convert("RGB")
    width, height = rgb.size
    scale = 1.0
    if cfg.max_side and max(width, height) > int(cfg.max_side):
        scale = int(cfg.max_side) / float(max(width, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        arr = np.asarray(rgb.resize((new_w, new_h)))
    else:
        arr = np.asarray(rgb)

    generator, device = _build_sam_mask_generator(cfg)
    try:
        masks = list(generator.generate(arr))
    except Exception:
        if device == "cpu":
            raise
        # Some ops (notably on MPS) can be unsupported; retry once on CPU.
        cpu_generator, _ = _build_sam_mask_generator(replace(cfg, device="cpu"))
        masks = list(cpu_generator.generate(arr))

    inv = 1.0 / scale if scale else 1.0
    raw: list[dict] = []
    for m in masks:
        bb = m.get("bbox") if isinstance(m, dict) else None
        if bb is not None and len(bb) == 4:
            x, y, bw, bh = (float(v) for v in bb)
        else:
            seg = np.asarray(m.get("segmentation")) if isinstance(m, dict) else None
            if seg is None or seg.ndim != 2 or not seg.any():
                continue
            ys, xs = np.where(seg)
            x, y, bw, bh = float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
        raw.append({
            "bbox": [x * inv, y * inv, (x + bw) * inv, (y + bh) * inv],
            "predicted_iou": float(m.get("predicted_iou", 0.0)) if isinstance(m, dict) else 0.0,
            "stability_score": float(m.get("stability_score", 0.0)) if isinstance(m, dict) else 0.0,
        })
    return raw


def _iou_box(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    ab = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter / max(1, aa + ab - inter))


def _dedupe_boxes_by_iou(scored: list[tuple[float, BBox]], iou_thresh: float) -> list[BBox]:
    """Greedy NMS-style dedup: keep highest-scored boxes, drop near-duplicates."""

    ordered = sorted(scored, key=lambda kv: kv[0], reverse=True)
    kept: list[BBox] = []
    for _, box in ordered:
        if all(_iou_box(box, kb) < float(iou_thresh) for kb in kept):
            kept.append(box)
    return kept


def _postprocess_sam_candidates(raw: Iterable[dict], width: int, height: int, cfg: SamConfig) -> list[BBox]:
    """Filter (side/area), IoU-deduplicate, and top-K raw XYXY SAM candidates.

    Each ``raw`` dict has ``bbox`` (XYXY, original image coords), ``predicted_iou``
    and ``stability_score``. Boxes are ranked by a
    predicted_iou / stability_score / area heuristic and the top
    ``cfg.max_proposals`` are kept.
    """

    total = float(max(1, width * height))
    scored: list[tuple[float, BBox]] = []
    for c in raw:
        bb = c.get("bbox")
        if bb is None or len(bb) != 4:
            continue
        box = _clip_bbox((int(round(bb[0])), int(round(bb[1])), int(round(bb[2])), int(round(bb[3]))), width, height)
        bw_, bh_ = box[2] - box[0], box[3] - box[1]
        if bw_ < int(cfg.min_side) or bh_ < int(cfg.min_side):
            continue
        area = float(bw_ * bh_) / total
        if area < float(cfg.min_area_frac) or area > float(cfg.max_area_frac):
            continue
        score = float(c.get("predicted_iou", 0.0)) + float(c.get("stability_score", 0.0)) + min(1.0, area * 4.0)
        scored.append((score, box))
    kept = _dedupe_boxes_by_iou(scored, cfg.dedupe_iou)
    return kept[: max(1, int(cfg.max_proposals))]


# --------------------------------------------------------------------------- #
# SAM proposal cache (skip re-running SAM on already-seen images)
# --------------------------------------------------------------------------- #
def _image_content_hash(pil: Image.Image) -> str:
    arr = np.asarray(pil.convert("RGB"))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def sam_cache_key(cfg: SamConfig, image_hash: str, image_id: object = None) -> str:
    """Deterministic cache key for one image's SAM proposals.

    Includes image identity (id + content hash), model type, checkpoint name, and
    every setting that changes the produced boxes (points_per_side, the two
    thresholds, max_proposals, the resize/crop settings, and the filter knobs).
    """

    ckpt_name = Path(cfg.checkpoint_path).name
    parts = [
        f"id={image_id if image_id is not None else 'na'}",
        f"img={image_hash}",
        f"model={cfg.model_type}",
        f"ckpt={ckpt_name}",
        f"pps={int(cfg.points_per_side)}",
        f"piou={float(cfg.pred_iou_thresh):.4f}",
        f"stab={float(cfg.stability_score_thresh):.4f}",
        f"maxp={int(cfg.max_proposals)}",
        f"maxside={int(cfg.max_side)}",
        f"crop={int(cfg.crop_n_layers)}",
        f"area={float(cfg.min_area_frac):.4f}-{float(cfg.max_area_frac):.4f}",
        f"minside={int(cfg.min_side)}",
        f"dedupe={float(cfg.dedupe_iou):.3f}",
    ]
    return "|".join(parts)


def _sam_cache_path(cache_dir: str | Path, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"sam_{digest}.json"


def _load_cached_sam_boxes(cache_path: Path) -> list[BBox] | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [tuple(int(v) for v in b) for b in data.get("boxes", [])]
    except Exception:
        return None


def _save_cached_sam_boxes(cache_path: Path, key: str, boxes: list[BBox]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"key": key, "boxes": [list(b) for b in boxes], "n": len(boxes)}),
            encoding="utf-8",
        )
    except Exception:  # pragma: no cover - cache write is best-effort
        pass


def _filter_injected_boxes(pil: Image.Image, raw: Iterable[BBox], cfg: SamConfig) -> list[BBox]:
    """Apply the same side/area/IoU filtering to an injected proposer's boxes."""

    w, h = pil.size
    total = float(max(1, w * h))
    scored: list[tuple[float, BBox]] = []
    for box in raw:
        b = _clip_bbox(tuple(int(v) for v in box), w, h)
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw < int(cfg.min_side) or bh < int(cfg.min_side):
            continue
        area = float(bw * bh) / total
        if area < float(cfg.min_area_frac) or area > float(cfg.max_area_frac):
            continue
        scored.append((float(bw * bh), b))
    return _dedupe_boxes_by_iou(scored, cfg.dedupe_iou)[: max(1, int(cfg.max_proposals))]


def sam_boxes(
    image: Image.Image | np.ndarray,
    *,
    allow_download: bool = False,
    enable_real_sam: bool = False,
    max_boxes: int = 48,
    sam_proposer: Callable[[Image.Image], Iterable[BBox]] | None = None,
    sam_config: SamConfig | None = None,
    checkpoint_path: str = DEFAULT_SAM_CHECKPOINT,
    model_type: str = DEFAULT_SAM_MODEL_TYPE,
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.90,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.80,
    min_side: int = 8,
    device: str = "auto",
    dedupe_iou: float = 0.7,
    crop_n_layers: int = 0,
    max_side: int = 0,
    cache_dir: str | None = None,
    image_id: object = None,
    use_cache: bool = False,
    **_: object,
) -> ProposalSet:
    """Segment Anything (SAM) automatic-mask proposal generator.

    Behaviour:
    * If a ``sam_proposer`` callable is injected (test / custom wrapper), use it
      and apply the same side/area/IoU filtering.
    * Otherwise real SAM is **opt-in** via ``enable_real_sam=True`` (wired from the
      sweep's ``--include-sam``). This keeps the default behaviour — and the other
      pilot script — unchanged, with a clear ``skip_reason``.
    * With ``enable_real_sam=True`` it loads ``segment_anything.sam_model_registry``
      + ``SamAutomaticMaskGenerator`` from a local checkpoint. It returns
      ``available=False`` with a clear ``skip_reason`` if the package or the
      checkpoint is missing. It NEVER downloads weights.
    """

    pil = _as_pil(image)
    cfg = sam_config or SamConfig(
        checkpoint_path=str(checkpoint_path),
        model_type=str(model_type),
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        max_proposals=int(max_boxes),
        min_area_frac=float(min_area_frac),
        max_area_frac=float(max_area_frac),
        min_side=int(min_side),
        device=str(device),
        dedupe_iou=float(dedupe_iou),
        crop_n_layers=int(crop_n_layers),
        max_side=int(max_side),
    )

    # 1. Injected proposer (used by tests and custom SAM wrappers).
    if sam_proposer is not None:
        try:
            raw = list(sam_proposer(pil))
        except Exception as exc:  # pragma: no cover - defensive
            return ProposalSet(name="sam_boxes", proposal_type=SAM_TYPE, boxes=[], available=False,
                               skip_reason=f"injected sam_proposer raised: {type(exc).__name__}: {exc}")
        boxes = _filter_injected_boxes(pil, raw, cfg)
        return ProposalSet(name="sam_boxes", proposal_type=SAM_TYPE, boxes=boxes, available=True)

    # 2. Real SAM is opt-in so default callers and the other pilot stay unchanged.
    if not enable_real_sam:
        return ProposalSet(
            name="sam_boxes",
            proposal_type=SAM_TYPE,
            boxes=[],
            available=False,
            skip_reason="real SAM not enabled (pass enable_real_sam=True / --include-sam); skipping cleanly",
        )

    # 3. Cache lookup FIRST — a cache hit replays cached proposals without loading
    #    SAM at all (so --resume works fast and even if the checkpoint was removed).
    w, h = pil.size
    cache_path = None
    if use_cache and cache_dir:
        key = sam_cache_key(cfg, _image_content_hash(pil), image_id)
        cache_path = _sam_cache_path(cache_dir, key)
        if cache_path.exists():
            cached_boxes = _load_cached_sam_boxes(cache_path)
            if cached_boxes is not None:
                return ProposalSet(
                    name="sam_boxes", proposal_type=SAM_TYPE, boxes=cached_boxes,
                    available=True, cache_hit=True, gen_seconds=0.0,
                )

    # 4. Package present?
    if not _module_installed("segment_anything"):
        return ProposalSet(
            name="sam_boxes",
            proposal_type=SAM_TYPE,
            boxes=[],
            available=False,
            skip_reason="`segment_anything` not installed; `pip install segment-anything` to enable (no auto-install)",
        )

    # 5. Local checkpoint present? (We never download it from Python.)
    if not Path(cfg.checkpoint_path).exists():
        return ProposalSet(
            name="sam_boxes",
            proposal_type=SAM_TYPE,
            boxes=[],
            available=False,
            skip_reason=f"SAM checkpoint not found at '{cfg.checkpoint_path}' (no auto-download); skipping cleanly",
        )

    # 6. Load (cached model) + run, timing the generation.
    t0 = time.monotonic()
    try:
        raw = _run_sam(pil, cfg)
    except Exception as exc:  # pragma: no cover - environment dependent
        return ProposalSet(
            name="sam_boxes",
            proposal_type=SAM_TYPE,
            boxes=[],
            available=False,
            skip_reason=f"SAM load/generate failed: {type(exc).__name__}: {exc}",
        )
    boxes = _postprocess_sam_candidates(raw, w, h, cfg)
    gen_seconds = time.monotonic() - t0
    if cache_path is not None:
        _save_cached_sam_boxes(cache_path, sam_cache_key(cfg, _image_content_hash(pil), image_id), boxes)
    return ProposalSet(
        name="sam_boxes", proposal_type=SAM_TYPE, boxes=boxes,
        available=True, cache_hit=False, gen_seconds=gen_seconds,
    )


def dino_boxes(
    image: Image.Image | np.ndarray,
    *,
    allow_download: bool = False,
    max_boxes: int = 32,
    dino_proposer: Callable[[Image.Image], Iterable[BBox]] | None = None,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.8,
    **_: object,
) -> ProposalSet:
    """Optional GroundingDINO / DINO-style detector adapter.

    Mirrors :func:`sam_boxes`: uses an injected ``dino_proposer`` if provided, and
    otherwise returns ``available=False`` with a clear ``skip_reason`` when no
    GroundingDINO / DINO detector is installed. Never auto-downloads weights.
    """

    pil = _as_pil(image)
    if dino_proposer is not None:
        try:
            raw = list(dino_proposer(pil))
        except Exception as exc:  # pragma: no cover - defensive
            return ProposalSet(name="dino_boxes", proposal_type=DINO_TYPE, boxes=[], available=False,
                               skip_reason=f"injected dino_proposer raised: {type(exc).__name__}: {exc}")
        boxes = _filter_boxes(pil, raw, min_area_frac, max_area_frac)[: max(1, int(max_boxes))]
        return ProposalSet(name="dino_boxes", proposal_type=DINO_TYPE, boxes=boxes, available=True)

    installed = (
        _module_installed("groundingdino")
        or _module_installed("GroundingDINO")
        or _module_installed("dino")
    )
    if not installed:
        return ProposalSet(
            name="dino_boxes",
            proposal_type=DINO_TYPE,
            boxes=[],
            available=False,
            skip_reason="GroundingDINO/DINO not installed (no `groundingdino` module); skipping cleanly",
        )
    if not allow_download:
        return ProposalSet(
            name="dino_boxes",
            proposal_type=DINO_TYPE,
            boxes=[],
            available=False,
            skip_reason="GroundingDINO installed but --allow-download not set; skipping (no auto-download)",
        )
    return ProposalSet(
        name="dino_boxes",
        proposal_type=DINO_TYPE,
        boxes=[],
        available=False,
        skip_reason="GroundingDINO installed and downloads allowed, but no dino_proposer wired in this repo; pass dino_proposer to enable",
    )


def _filter_boxes(pil: Image.Image, raw: Iterable[BBox], min_area_frac: float, max_area_frac: float) -> list[BBox]:
    w, h = pil.size
    total = float(max(1, w * h))
    out: list[BBox] = []
    for box in raw:
        bbox = _clip_bbox(tuple(int(v) for v in box), w, h)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        area = float(bw * bh) / total
        if bw < 4 or bh < 4 or area < float(min_area_frac) or area > float(max_area_frac):
            continue
        out.append(bbox)
    return out


# --------------------------------------------------------------------------- #
# Registry / orchestration
# --------------------------------------------------------------------------- #
_GENERATORS: dict[str, Callable[..., ProposalSet]] = {
    "grid_boxes": grid_boxes,
    "edge_component_boxes": edge_component_boxes,
    "saliency_boxes": saliency_boxes,
    "sam_boxes": sam_boxes,
    "dino_boxes": dino_boxes,
}


def available_generators() -> list[str]:
    """Names of all registered generator families (available or not)."""

    return list(_GENERATORS.keys())


def generator_availability(*, allow_download: bool = False, **kwargs: object) -> dict[str, ProposalSet]:
    """Probe each generator on a tiny synthetic image to report availability.

    Returns a mapping ``name -> ProposalSet`` (with ``available`` / ``skip_reason``)
    without requiring real data, so the experiment summaries can list which
    generators are available and which were skipped (and why).
    """

    probe = Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8))
    out: dict[str, ProposalSet] = {}
    for name, fn in _GENERATORS.items():
        try:
            out[name] = fn(probe, allow_download=allow_download, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            out[name] = ProposalSet(
                name=name, proposal_type=f"auto_{name}", boxes=[], available=False,
                skip_reason=f"probe raised {type(exc).__name__}: {exc}",
            )
    return out


def generate_proposal_sets(
    image: Image.Image | np.ndarray,
    families: Iterable[str] | None = None,
    *,
    allow_download: bool = False,
    max_boxes: int = 48,
    sam_proposer: Callable[[Image.Image], Iterable[BBox]] | None = None,
    dino_proposer: Callable[[Image.Image], Iterable[BBox]] | None = None,
    **kwargs: object,
) -> dict[str, ProposalSet]:
    """Run the requested generator families on one image."""

    fams = list(families) if families is not None else list(ALL_FAMILIES)
    out: dict[str, ProposalSet] = {}
    for name in fams:
        fn = _GENERATORS.get(name)
        if fn is None:
            out[name] = ProposalSet(name=name, proposal_type="unknown", boxes=[], available=False,
                                    skip_reason=f"unknown generator family '{name}'")
            continue
        extra: dict[str, object] = {"max_boxes": max_boxes, "allow_download": allow_download}
        if name == "sam_boxes":
            extra["sam_proposer"] = sam_proposer
        if name == "dino_boxes":
            extra["dino_proposer"] = dino_proposer
        extra.update(kwargs)
        out[name] = fn(image, **extra)
    return out


def proposal_sets_to_region_proposals(
    image: Image.Image | np.ndarray,
    proposal_sets: Iterable[ProposalSet],
    *,
    include_random_control: bool = True,
    seed: int = 0,
    n_random: int = 24,
) -> list[RegionProposal]:
    """Convert generated boxes into ``RegionProposal`` objects for CIC scoring.

    Random control patches are appended (unless disabled) so a matched-random
    baseline is always available downstream. Features are derived from pixels and
    geometry only — no labels enter here.
    """

    pil = _as_pil(image)
    edges = _gray_edges(pil)
    out: list[RegionProposal] = []
    for ps in proposal_sets:
        if not ps.available:
            continue
        for i, box in enumerate(ps.boxes):
            out.append(proposal_from_bbox(pil, tuple(int(v) for v in box), f"{ps.name}_{i:04d}", ps.proposal_type))
    if include_random_control:
        for r in random_patch_proposals(pil, edges, seed=seed, n=n_random):
            out.append(
                RegionProposal(
                    candidate_id=r.candidate_id,
                    bbox=r.bbox,
                    proposal_type=RANDOM_TYPE,
                    area_fraction=r.area_fraction,
                    edge_density=r.edge_density,
                    horizontalness_score=r.horizontalness_score,
                    center_overlap_score=r.center_overlap_score,
                    border_distance=r.border_distance,
                    center_distance=r.center_distance,
                    width=r.width,
                    height=r.height,
                    aspect_ratio=r.aspect_ratio,
                )
            )
    return _dedupe(out)
