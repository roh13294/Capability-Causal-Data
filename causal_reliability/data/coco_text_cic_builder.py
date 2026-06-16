from __future__ import annotations

"""Build a curated COCO-Text x COCO-objects metadata CSV for the natural-text CIC.

This module joins the locally-present COCO 2014 *train* object annotations
(``instances_train2014.json``) with the COCO-Text v2 scene-text annotations
(``cocotext.v2.json``) by ``image_id`` and emits a small, filtered metadata CSV
compatible with the ``local`` adapter in
:mod:`causal_reliability.data.natural_text_dataset`.

Each emitted row describes one image that contains BOTH:

* a clearly dominant object from an allowed CLIP-friendly category (the visual
  ``human_label`` / content), and
* at least one legible (English) scene-text box that is reasonably separable
  from that object (a candidate text *shortcut* region).

The CSV schema matches the ``local`` folder mode exactly::

    image_path, human_label, allowed_clip_labels,
    optional_text_boxes, optional_object_boxes, source, notes

Box columns are JSON ``[[x0,y0,x1,y1], ...]`` lists in *original pixel*
coordinates (the loader rescales them). Box metadata is evaluation / oracle-only;
the candidate scoring rule never sees it.

IMPORTANT: this builder reads *local* files only — it never downloads. It also
only writes ``data/coco_text_cic/metadata.csv`` (and nothing under ``results/``),
so it cannot disturb any final-report metric or curated Round-1 artifact. It does
NOT run CLIP/CIC.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in original pixels


# --------------------------------------------------------------------------- #
# Allowed object categories (CLIP-friendly, single-dominant-object friendly).
# "person" is intentionally excluded: it is ubiquitous and rarely a single,
# isolable dominant object. The set is configurable via ``FilterConfig``.
# --------------------------------------------------------------------------- #
DEFAULT_ALLOWED_CATEGORIES: tuple[str, ...] = (
    # animals
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    # food
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    # common objects
    "bottle", "cup", "laptop", "cell phone", "clock", "vase", "teddy bear", "umbrella",
    "backpack", "suitcase",
)


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for the join + filtering pass.

    Fractional thresholds are relative to the full image area / dimension. Box
    coordinates are original pixels.
    """

    allowed_categories: tuple[str, ...] = DEFAULT_ALLOWED_CATEGORIES
    require_legible: bool = True
    require_english: bool = True
    # Object (content) box must be a meaningful fraction of the image.
    min_object_area_frac: float = 0.05
    # Text box must be large enough to be a usable region (frac of image area)
    # and at least a few px on each side.
    min_text_area_frac: float = 0.002
    min_text_box_px: int = 8
    # A text box is rejected if too much of it sits inside the dominant object
    # box (we want text *separable* from the content, not painted on it).
    max_text_object_overlap: float = 0.5
    # "Not too many dominant objects": count allowed-category objects whose area
    # fraction is >= this, and require the count to be <= ``max_dominant_objects``.
    dominant_object_area_frac: float = 0.10
    max_dominant_objects: int = 2

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "FilterConfig":
        cfg = dict(cfg or {})
        cats = cfg.get("allowed_categories")
        kwargs: dict[str, Any] = {}
        if cats:
            kwargs["allowed_categories"] = tuple(str(c).strip() for c in cats if str(c).strip())
        for fname in (
            "require_legible", "require_english", "min_object_area_frac",
            "min_text_area_frac", "min_text_box_px", "max_text_object_overlap",
            "dominant_object_area_frac", "max_dominant_objects",
        ):
            if fname in cfg and cfg[fname] is not None:
                kwargs[fname] = cfg[fname]
        return cls(**kwargs)


@dataclass
class BuildStats:
    coco_text_images_loaded: int = 0
    coco_text_val_images: int = 0
    train2014_images_found: int = 0
    instances_images_loaded: int = 0
    ids_with_both: int = 0
    ids_after_filtering: int = 0
    schema_issues: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Path detection
# --------------------------------------------------------------------------- #
def detect_paths(raw_root: str | Path) -> dict[str, Path | None]:
    """Locate the four required inputs under ``raw_root`` (or its parents).

    Returns a dict with keys ``train_dir``, ``instances``, ``captions``,
    ``cocotext``; missing items are ``None``.
    """

    raw_root = Path(raw_root)
    search_roots = [raw_root, raw_root / "raw", raw_root.parent]

    def _first(rel_candidates: list[str]) -> Path | None:
        for base in search_roots:
            for rel in rel_candidates:
                p = base / rel
                if p.exists():
                    return p
        return None

    train_dir = _first(["train2014", "images/train2014", "raw/train2014"])
    instances = _first([
        "annotations/instances_train2014.json",
        "instances_train2014.json",
        "raw/annotations/instances_train2014.json",
    ])
    captions = _first([
        "annotations/captions_train2014.json",
        "captions_train2014.json",
        "raw/annotations/captions_train2014.json",
    ])
    cocotext = _first(["cocotext.v2.json", "annotations/cocotext.v2.json", "raw/cocotext.v2.json"])
    return {"train_dir": train_dir, "instances": instances, "captions": captions, "cocotext": cocotext}


# --------------------------------------------------------------------------- #
# Geometry helpers (original-pixel coordinates)
# --------------------------------------------------------------------------- #
def xywh_to_xyxy(box: list[float]) -> BBox:
    x, y, w, h = box[:4]
    return (int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)))


def box_area(b: BBox) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection_area(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def overlap_fraction_of_first(a: BBox, b: BBox) -> float:
    """Fraction of box ``a``'s area covered by box ``b``."""

    area = box_area(a)
    if area <= 0:
        return 0.0
    return intersection_area(a, b) / area


# --------------------------------------------------------------------------- #
# Loaders (local only)
# --------------------------------------------------------------------------- #
def load_coco_text(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (imgs_by_id_str, anns_by_id_str) from cocotext.v2.json."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("imgs", {}), payload.get("anns", {})


def load_instances(path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    """Return (images_by_id, object_anns_by_image_id, category_id_to_name)."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images_by_id = {int(im["id"]): im for im in payload.get("images", [])}
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in payload.get("categories", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    return images_by_id, anns_by_image, cat_id_to_name


def load_captions(path: str | Path | None) -> dict[int, str]:
    """Return image_id -> first caption (best-effort; empty if unavailable)."""

    if not path or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out: dict[int, str] = {}
    for ann in payload.get("annotations", []):
        iid = int(ann["image_id"])
        if iid not in out:
            out[iid] = str(ann.get("caption", "")).strip()
    return out


# --------------------------------------------------------------------------- #
# Per-image evaluation
# --------------------------------------------------------------------------- #
def _qualifying_text_boxes(
    text_anns: list[dict[str, Any]],
    image_area: float,
    cfg: FilterConfig,
) -> list[BBox]:
    """Legible (English) text boxes that are large enough to be usable."""

    out: list[BBox] = []
    for ann in text_anns:
        if cfg.require_legible and ann.get("legibility") != "legible":
            continue
        if cfg.require_english and ann.get("language") not in ("english", None, ""):
            continue
        raw = ann.get("bbox")
        if not raw or len(raw) < 4:
            continue
        box = xywh_to_xyxy(raw)
        w, h = box[2] - box[0], box[3] - box[1]
        if w < cfg.min_text_box_px or h < cfg.min_text_box_px:
            continue
        if image_area > 0 and box_area(box) / image_area < cfg.min_text_area_frac:
            continue
        out.append(box)
    return out


def _allowed_object_boxes(
    object_anns: list[dict[str, Any]],
    cat_id_to_name: dict[int, str],
    image_area: float,
    cfg: FilterConfig,
) -> list[tuple[str, BBox, float]]:
    """Allowed-category object boxes above the min area threshold.

    Returns list of (category_name, box, area_fraction), largest first.
    """

    allowed = set(cfg.allowed_categories)
    out: list[tuple[str, BBox, float]] = []
    for ann in object_anns:
        if ann.get("iscrowd"):
            continue
        name = cat_id_to_name.get(int(ann.get("category_id", -1)))
        if name not in allowed:
            continue
        raw = ann.get("bbox")
        if not raw or len(raw) < 4:
            continue
        box = xywh_to_xyxy(raw)
        frac = (box_area(box) / image_area) if image_area > 0 else 0.0
        if frac < cfg.min_object_area_frac:
            continue
        out.append((name, box, frac))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def evaluate_image(
    image_id: int,
    coco_image_meta: dict[str, Any],
    object_anns: list[dict[str, Any]],
    text_anns: list[dict[str, Any]],
    cat_id_to_name: dict[int, str],
    cfg: FilterConfig,
) -> dict[str, Any] | None:
    """Apply all filters to one joined image. Returns a row dict or ``None``."""

    width = int(coco_image_meta.get("width", 0))
    height = int(coco_image_meta.get("height", 0))
    image_area = float(width * height)

    allowed_objs = _allowed_object_boxes(object_anns, cat_id_to_name, image_area, cfg)
    if not allowed_objs:
        return None  # no large allowed-category object

    # Not too many dominant objects.
    n_dominant = sum(1 for _, _, frac in allowed_objs if frac >= cfg.dominant_object_area_frac)
    if n_dominant > cfg.max_dominant_objects:
        return None

    dominant_name, dominant_box, dominant_frac = allowed_objs[0]

    text_boxes = _qualifying_text_boxes(text_anns, image_area, cfg)
    if not text_boxes:
        return None  # no usable legible text

    # Keep only text boxes that are separable from the dominant object.
    separable_text = [
        tb for tb in text_boxes
        if overlap_fraction_of_first(tb, dominant_box) <= cfg.max_text_object_overlap
    ]
    if not separable_text:
        return None

    object_boxes = [box for _, box, _ in allowed_objs]
    return {
        "image_id": image_id,
        "file_name": coco_image_meta.get("file_name", f"COCO_train2014_{image_id:012d}.jpg"),
        "human_label": dominant_name,
        "dominant_area_frac": round(dominant_frac, 4),
        "n_dominant_objects": n_dominant,
        "object_boxes": object_boxes,
        "text_boxes": separable_text,
        "n_text_boxes": len(separable_text),
    }


# --------------------------------------------------------------------------- #
# Top-level build
# --------------------------------------------------------------------------- #
def build_metadata(
    paths: dict[str, Path | None],
    cfg: FilterConfig,
    max_images: int | None,
    prefer_set: str = "val",
    image_path_prefix: str = "raw/train2014",
    captions: dict[int, str] | None = None,
) -> tuple[list[dict[str, Any]], BuildStats]:
    """Join, filter, and assemble metadata rows.

    Returns (rows, stats). Rows use the ``local`` CSV schema. ``image_path`` is
    written relative to the dataset root as ``{image_path_prefix}/{file_name}``.
    """

    stats = BuildStats()
    train_dir = paths.get("train_dir")
    instances_path = paths.get("instances")
    cocotext_path = paths.get("cocotext")

    if instances_path is None:
        stats.schema_issues.append("instances_train2014.json not found")
        return [], stats
    if cocotext_path is None:
        stats.schema_issues.append("cocotext.v2.json not found")
        return [], stats

    ct_imgs, ct_anns = load_coco_text(cocotext_path)
    stats.coco_text_images_loaded = len(ct_imgs)
    stats.coco_text_val_images = sum(1 for v in ct_imgs.values() if v.get("set") == "val")

    images_by_id, obj_anns_by_image, cat_id_to_name = load_instances(instances_path)
    stats.instances_images_loaded = len(images_by_id)

    # Index COCO-Text annotations by image_id.
    ct_anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in ct_anns.values():
        ct_anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    # train2014 image files actually present on disk.
    present_files: set[str] = set()
    if train_dir is not None and train_dir.exists():
        present_files = {p.name for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        stats.train2014_images_found = len(present_files)
    else:
        stats.schema_issues.append("train2014 image directory not found")

    # Join: image ids present in BOTH COCO objects and COCO-Text.
    both_ids = set(images_by_id) & set(ct_anns_by_image)
    stats.ids_with_both = len(both_ids)

    # Order so COCO-Text set == prefer_set ("val") comes first, then by id for
    # determinism. Images still load from train2014/ regardless of set.
    def _sort_key(iid: int) -> tuple[int, int]:
        ct_meta = ct_imgs.get(str(iid), {})
        is_preferred = 0 if ct_meta.get("set") == prefer_set else 1
        return (is_preferred, iid)

    rows: list[dict[str, Any]] = []
    allowed_vocab = "|".join(sorted(set(cfg.allowed_categories)))
    captions = captions or {}

    for iid in sorted(both_ids, key=_sort_key):
        if max_images is not None and len(rows) >= max_images:
            break
        coco_meta = images_by_id[iid]
        file_name = coco_meta.get("file_name", f"COCO_train2014_{iid:012d}.jpg")
        # Require the image to be physically present (so the loader can read it).
        if present_files and file_name not in present_files:
            continue
        result = evaluate_image(
            iid,
            coco_meta,
            obj_anns_by_image.get(iid, []),
            ct_anns_by_image.get(iid, []),
            cat_id_to_name,
            cfg,
        )
        if result is None:
            continue
        ct_set = ct_imgs.get(str(iid), {}).get("set", "")
        caption = captions.get(iid, "")
        note_bits = [f"coco_text_set={ct_set}", f"dominant_area_frac={result['dominant_area_frac']}",
                     f"n_text_boxes={result['n_text_boxes']}"]
        if caption:
            note_bits.append(f"caption={caption}")
        rows.append({
            "image_path": f"{image_path_prefix}/{file_name}",
            "human_label": result["human_label"],
            "allowed_clip_labels": allowed_vocab,
            "optional_text_boxes": json.dumps([list(b) for b in result["text_boxes"]]),
            "optional_object_boxes": json.dumps([list(b) for b in result["object_boxes"]]),
            "source": "coco_text_cic",
            "notes": "; ".join(note_bits),
        })

    stats.ids_after_filtering = len(rows)
    return rows, stats
