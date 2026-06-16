from __future__ import annotations

"""Natural-image dataset adapters for the open-candidate (shortcut-agnostic) CIC.

This module supplies images that contain *real-world* scene text, signage, or
meme-style captions rather than synthetically rendered overlays. It supports
three loading modes:

1. ``local`` folder mode: a directory of images plus a metadata CSV with the
   columns ``image_path, human_label, allowed_clip_labels, optional_text_boxes,
   optional_object_boxes, source, notes``.
2. public-dataset metadata adapters (TextOCR / COCO-Text / Open Images style).
   These read *locally present* annotation files only; they never download.
3. ``synthetic`` mode that renders a small "natural-like" fixture so tests and
   smoke runs do not require any external data.

Boxes carried on an example (``text_boxes``/``object_boxes``) are evaluation and
oracle-upper-bound metadata. The candidate *scoring* rule never receives the true
label, the shortcut identity, the OCR text content, or correctness — see
``causal_reliability.discovery.open_region_proposals`` and
``causal_reliability.discovery.cic_region_scoring``.
"""

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from causal_reliability.utils.io import ensure_dir


BBox = tuple[int, int, int, int]

# A compact, natural-image vocabulary used by the synthetic fixture. Real data
# supplies its own ``allowed_clip_labels`` per image.
SYNTHETIC_OBJECT_LABELS = ["dog", "cat", "car", "bird", "flower", "cup"]
SYNTHETIC_SCENE_WORDS = ["SALE", "STOP", "PIZZA", "OPEN", "FREE", "TAXI"]


@dataclass(frozen=True)
class NaturalTextExample:
    example_id: int
    image: np.ndarray  # float32 (H, W, 3) in [0, 1]
    human_label: str
    label: int  # index into allowed_clip_labels
    allowed_clip_labels: list[str]
    text_boxes: list[BBox]  # eval-only / OCR-detector-like candidate source
    object_boxes: list[BBox]  # eval-only content-preservation reference
    source: str
    notes: str
    split: str = "test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "image": self.image,
            "human_label": self.human_label,
            "label": self.label,
            "allowed_clip_labels": list(self.allowed_clip_labels),
            "text_boxes": [list(b) for b in self.text_boxes],
            "object_boxes": [list(b) for b in self.object_boxes],
            "source": self.source,
            "notes": self.notes,
            "split": self.split,
        }


@dataclass(frozen=True)
class NaturalTextBundle:
    examples: list[dict[str, Any]]
    label_vocabulary: list[str]
    mode: str
    notes: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_label_list(value: Any) -> list[str]:
    """Parse ``allowed_clip_labels`` from a CSV cell.

    Accepts pipe-, semicolon-, or comma-separated strings, or a JSON list.
    """

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
    for sep in ("|", ";"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def parse_bbox(value: Any) -> BBox | None:
    """Parse a single bbox into an ``(x0, y0, x1, y1)`` int tuple."""

    if value is None:
        return None
    seq: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            seq = ast.literal_eval(text)
        except Exception:
            cleaned = text.replace("[", " ").replace("]", " ").replace(",", " ")
            parts = [p for p in cleaned.split() if p]
            try:
                seq = [float(p) for p in parts]
            except Exception:
                return None
    try:
        vals = [int(round(float(v))) for v in seq]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    x0, y0, x1, y1 = vals
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def parse_bbox_list(value: Any) -> list[BBox]:
    """Parse a list of bboxes from a CSV cell.

    Accepts ``[[x0,y0,x1,y1], ...]`` JSON, a single bbox, or empty.
    """

    if value is None:
        return []
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        out = [parse_bbox(b) for b in value]
        return [b for b in out if b is not None]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        single = parse_bbox(text)
        return [single] if single is not None else []
    if isinstance(parsed, (list, tuple)) and parsed and isinstance(parsed[0], (list, tuple)):
        out = [parse_bbox(b) for b in parsed]
        return [b for b in out if b is not None]
    single = parse_bbox(parsed)
    return [single] if single is not None else []


def _load_image_array(path: Path, image_size: int | None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if image_size is not None and image_size > 0:
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def _scale_boxes(boxes: list[BBox], orig_size: tuple[int, int], image_size: int | None) -> list[BBox]:
    if image_size is None or image_size <= 0:
        return boxes
    ow, oh = orig_size
    sx, sy = image_size / max(1, ow), image_size / max(1, oh)
    out: list[BBox] = []
    for x0, y0, x1, y1 in boxes:
        out.append((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
    return out


# --------------------------------------------------------------------------- #
# Local folder mode
# --------------------------------------------------------------------------- #
def load_local_folder_dataset(
    root: str | Path,
    metadata_csv: str | Path | None = None,
    image_size: int | None = 224,
    split: str = "test",
) -> NaturalTextBundle:
    """Load images + metadata CSV from a local folder.

    Returns an empty bundle (rather than raising) when the folder or CSV is
    missing, so the experiment runner can degrade gracefully.
    """

    import pandas as pd

    root = Path(root)
    csv_path = Path(metadata_csv) if metadata_csv else root / "metadata.csv"
    if not csv_path.exists():
        return NaturalTextBundle([], [], "local", notes=f"metadata CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    examples: list[dict[str, Any]] = []
    vocab: set[str] = set()
    for i, record in frame.iterrows():
        rel = str(record.get("image_path", "")).strip()
        if not rel:
            continue
        image_path = Path(rel)
        if not image_path.is_absolute():
            image_path = root / rel
        if not image_path.exists():
            continue
        human_label = str(record.get("human_label", "")).strip()
        allowed = parse_label_list(record.get("allowed_clip_labels"))
        if human_label and human_label not in allowed:
            allowed = [human_label, *allowed]
        if not allowed:
            continue
        with Image.open(image_path) as probe:
            orig_size = probe.size
        text_boxes = _scale_boxes(parse_bbox_list(record.get("optional_text_boxes")), orig_size, image_size)
        object_boxes = _scale_boxes(parse_bbox_list(record.get("optional_object_boxes")), orig_size, image_size)
        label = allowed.index(human_label) if human_label in allowed else 0
        examples.append(
            NaturalTextExample(
                example_id=int(i),
                image=_load_image_array(image_path, image_size),
                human_label=human_label or allowed[label],
                label=label,
                allowed_clip_labels=allowed,
                text_boxes=text_boxes,
                object_boxes=object_boxes,
                source=str(record.get("source", "")).strip() or "local_folder",
                notes=str(record.get("notes", "")).strip(),
                split=split,
            ).to_dict()
        )
        vocab.update(allowed)
    return NaturalTextBundle(examples, sorted(vocab), "local", notes=f"loaded {len(examples)} local images")


# --------------------------------------------------------------------------- #
# Verified natural-text failure annotations
# --------------------------------------------------------------------------- #
def parse_pipe_bbox_list(value: Any) -> list[BBox]:
    """Parse a ``|``-separated list of ``x0,y0,x1,y1`` boxes from a CSV cell.

    This is the format used by ``verified_annotations.csv`` (e.g.
    ``"63,1,446,52|1,413,493,500"``). Each segment is parsed with
    :func:`parse_bbox`; empty/blank cells yield an empty list.
    """

    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    out: list[BBox] = []
    for part in text.split("|"):
        box = parse_bbox(part)
        if box is not None:
            out.append(box)
    return out


def load_verified_natural_text_dataset(
    root: str | Path,
    annotations_csv: str | Path | None = None,
    image_size: int | None = 224,
    split: str = "test",
    include_only: bool = True,
) -> NaturalTextBundle:
    """Load the human-verified natural-text failure annotation set.

    Reads ``verified_annotations.csv`` whose schema is::

        image_path, visual_target_label, visual_label_aliases,
        text_distractor_labels, text_or_logo_boxes, object_boxes,
        text_driven_candidate, include_in_verified_failure_eval,
        exclusion_reason, notes

    ``visual_label_aliases`` is the full candidate label set shown to CLIP; the
    correct index is the position of ``visual_target_label`` within it.
    ``text_distractor_labels`` are the non-target labels whose selection counts
    as a (text-driven) failure. Box columns are in original-pixel coordinates and
    are rescaled to ``image_size`` (eval/oracle-only metadata; never seen by the
    candidate scoring rule).

    When ``include_only`` is True (default), only rows with
    ``include_in_verified_failure_eval == yes`` are returned. Returns an empty
    bundle if the CSV is missing.
    """

    import pandas as pd

    root = Path(root)
    csv_path = Path(annotations_csv) if annotations_csv else root / "verified_annotations.csv"
    if not csv_path.exists():
        return NaturalTextBundle([], [], "verified_local", notes=f"verified annotations CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    n_total = int(len(frame))
    examples: list[dict[str, Any]] = []
    vocab: set[str] = set()
    n_included = 0
    for i, record in frame.iterrows():
        include_flag = str(record.get("include_in_verified_failure_eval", "")).strip().lower() == "yes"
        if include_flag:
            n_included += 1
        if include_only and not include_flag:
            continue
        rel = str(record.get("image_path", "")).strip()
        if not rel:
            continue
        image_path = Path(rel)
        if not image_path.is_absolute():
            image_path = root / rel
        if not image_path.exists():
            continue
        target = str(record.get("visual_target_label", "")).strip()
        allowed = parse_label_list(record.get("visual_label_aliases"))
        if target and target not in allowed:
            allowed = [target, *allowed]
        if not allowed:
            continue
        distractors = [d for d in parse_label_list(record.get("text_distractor_labels")) if d in allowed]
        with Image.open(image_path) as probe:
            orig_size = probe.size
        text_boxes = _scale_boxes(parse_pipe_bbox_list(record.get("text_or_logo_boxes")), orig_size, image_size)
        object_boxes = _scale_boxes(parse_pipe_bbox_list(record.get("object_boxes")), orig_size, image_size)
        label = allowed.index(target) if target in allowed else 0
        ex = NaturalTextExample(
            example_id=int(i),
            image=_load_image_array(image_path, image_size),
            human_label=target or allowed[label],
            label=label,
            allowed_clip_labels=allowed,
            text_boxes=text_boxes,
            object_boxes=object_boxes,
            source="verified_natural_text",
            notes=str(record.get("notes", "")).strip(),
            split=split,
        ).to_dict()
        # Extra verified-failure metadata (eval-only): which candidate labels are
        # text/logo distractors, and whether the row was human-flagged as a
        # text-driven candidate.
        ex["text_distractor_labels"] = distractors
        ex["text_driven_candidate"] = str(record.get("text_driven_candidate", "")).strip().lower()
        ex["include_in_verified_failure_eval"] = bool(include_flag)
        examples.append(ex)
        vocab.update(allowed)
    notes = (
        f"verified annotations: loaded {len(examples)} of {n_included} include=yes "
        f"rows ({n_total} total) from {csv_path.name}"
    )
    bundle = NaturalTextBundle(examples, sorted(vocab), "verified_local", notes=notes)
    object.__setattr__(bundle, "diagnostics", {"n_total_rows": n_total, "n_include_yes": n_included})
    return bundle


# --------------------------------------------------------------------------- #
# Public-dataset metadata adapters (read local files only; never download)
# --------------------------------------------------------------------------- #
def _public_adapter(
    metadata_json: str | Path | None,
    image_root: str | Path | None,
    image_size: int | None,
    split: str,
    source_name: str,
    box_extractor,
) -> NaturalTextBundle:
    if not metadata_json:
        return NaturalTextBundle([], [], source_name, notes=f"{source_name}: no metadata path configured")
    meta_path = Path(metadata_json)
    if not meta_path.exists():
        return NaturalTextBundle([], [], source_name, notes=f"{source_name}: metadata not found at {meta_path}")
    root = Path(image_root) if image_root else meta_path.parent
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = box_extractor(payload)
    examples: list[dict[str, Any]] = []
    vocab: set[str] = set()
    for i, rec in enumerate(records):
        image_path = root / rec["image_file"]
        if not image_path.exists():
            continue
        allowed = rec.get("allowed_clip_labels") or []
        human_label = rec.get("human_label", "")
        if human_label and human_label not in allowed:
            allowed = [human_label, *allowed]
        if not allowed:
            continue
        with Image.open(image_path) as probe:
            orig_size = probe.size
        label = allowed.index(human_label) if human_label in allowed else 0
        examples.append(
            NaturalTextExample(
                example_id=i,
                image=_load_image_array(image_path, image_size),
                human_label=human_label or allowed[label],
                label=label,
                allowed_clip_labels=list(allowed),
                text_boxes=_scale_boxes(rec.get("text_boxes", []), orig_size, image_size),
                object_boxes=_scale_boxes(rec.get("object_boxes", []), orig_size, image_size),
                source=source_name,
                notes=rec.get("notes", ""),
                split=split,
            ).to_dict()
        )
        vocab.update(allowed)
    return NaturalTextBundle(examples, sorted(vocab), source_name, notes=f"{source_name}: loaded {len(examples)} images")


def _textocr_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapter stub for TextOCR-style metadata.

    Expects a pre-digested ``images`` list of ``{image_file, human_label,
    allowed_clip_labels, text_boxes, object_boxes, notes}``. TextOCR's raw
    schema can be projected into this shape offline; we do not vendor the full
    TextOCR loader here.
    """

    return list(payload.get("images", []))


def _coco_text_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapter stub for COCO-Text-style metadata (same digested shape)."""

    return list(payload.get("images", []))


def _open_images_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapter stub for Open Images-style metadata (same digested shape)."""

    return list(payload.get("images", []))


def load_textocr_metadata(metadata_json, image_root=None, image_size=224, split="test") -> NaturalTextBundle:
    return _public_adapter(metadata_json, image_root, image_size, split, "textocr", _textocr_records)


def load_coco_text_metadata(metadata_json, image_root=None, image_size=224, split="test") -> NaturalTextBundle:
    return _public_adapter(metadata_json, image_root, image_size, split, "coco_text", _coco_text_records)


def load_open_images_metadata(metadata_json, image_root=None, image_size=224, split="test") -> NaturalTextBundle:
    return _public_adapter(metadata_json, image_root, image_size, split, "open_images", _open_images_records)


# --------------------------------------------------------------------------- #
# Synthetic "natural-like" fixture
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", max(10, size // 8))
    except Exception:
        return ImageFont.load_default()


def _render_natural_like(
    size: int,
    object_label_index: int,
    scene_word: str,
    misleading: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, BBox, BBox]:
    """Render a noisy, gradient-lit scene with a central object and scene text.

    Returns (image_array, object_box, text_box). This is a stand-in for natural
    photographs; it is *not* used as headline evidence — only as a fixture so the
    pipeline can run end-to-end without external data.
    """

    base = np.zeros((size, size, 3), dtype=np.float32)
    # Gradient background with per-channel tilt + noise to look photo-like.
    yy = np.linspace(0.0, 1.0, size)[:, None]
    xx = np.linspace(0.0, 1.0, size)[None, :]
    tilt = rng.uniform(-0.3, 0.3, size=3)
    tint = rng.uniform(0.35, 0.75, size=3)
    for c in range(3):
        base[:, :, c] = tint[c] + tilt[c] * (0.5 * yy + 0.5 * xx)
    base += rng.normal(0.0, 0.05, size=base.shape).astype(np.float32)
    img = Image.fromarray((base.clip(0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Central object whose appearance is tied to the (true) object label.
    box_w = int(size * rng.uniform(0.34, 0.46))
    box_h = int(size * rng.uniform(0.34, 0.46))
    ox = int((size - box_w) * rng.uniform(0.25, 0.75))
    oy = int((size - box_h) * rng.uniform(0.12, 0.45))
    object_box: BBox = (ox, oy, ox + box_w, oy + box_h)
    palette = [
        (196, 64, 48),
        (60, 120, 200),
        (70, 160, 90),
        (210, 170, 50),
        (150, 80, 170),
        (90, 90, 90),
    ]
    color = palette[object_label_index % len(palette)]
    if object_label_index % 2 == 0:
        draw.ellipse(list(object_box), fill=color)
    else:
        draw.rectangle(list(object_box), fill=color)

    # Scene text / sign placed away from the object so a text-removal repair does
    # not necessarily damage the object region.
    font = _font(size)
    tb = draw.textbbox((0, 0), scene_word, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = max(2, min(size - tw - 2, int(size * rng.uniform(0.04, 0.5))))
    ty = int(size * 0.80)
    pad = max(3, size // 40)
    text_box: BBox = (tx - pad, ty - pad, tx + tw + pad, ty + th + pad)
    panel = (245, 245, 245) if not misleading else (250, 240, 235)
    draw.rectangle(list(text_box), fill=panel)
    draw.text((tx, ty), scene_word, font=font, fill=(170, 25, 30))

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, object_box, text_box


def make_synthetic_natural_text_dataset(
    n_images: int = 12,
    size: int = 96,
    seed: int = 0,
    split: str = "test",
) -> NaturalTextBundle:
    """Generate a tiny natural-like fixture for smoke tests and pipeline checks."""

    rng = np.random.default_rng(seed)
    examples: list[dict[str, Any]] = []
    vocab: set[str] = set()
    n_labels = len(SYNTHETIC_OBJECT_LABELS)
    for i in range(n_images):
        obj_idx = i % n_labels
        human_label = SYNTHETIC_OBJECT_LABELS[obj_idx]
        # A misleading scene word points at a different label (typographic-style
        # shortcut), present on most but not all images.
        misleading = (i % 4) != 0
        # A distinct distractor label always present so the choice is non-trivial.
        distractor_idx = (obj_idx + 1 + (i % (n_labels - 1))) % n_labels
        # When misleading, the scene word points at the distractor; otherwise it
        # names the true object (an aligned, non-harmful caption).
        word_idx = distractor_idx if misleading else obj_idx
        scene_word = SYNTHETIC_SCENE_WORDS[word_idx]
        distractor = SYNTHETIC_OBJECT_LABELS[distractor_idx]
        allowed = sorted({human_label, distractor})
        label = allowed.index(human_label)
        image, object_box, text_box = _render_natural_like(size, obj_idx, scene_word, misleading, rng)
        examples.append(
            NaturalTextExample(
                example_id=i,
                image=image,
                human_label=human_label,
                label=label,
                allowed_clip_labels=allowed,
                text_boxes=[text_box],
                object_boxes=[object_box],
                source="synthetic_natural_like",
                notes=f"misleading_scene_text={misleading}; word={scene_word}",
                split=split,
            ).to_dict()
        )
        vocab.update(allowed)
    return NaturalTextBundle(examples, sorted(vocab), "synthetic", notes=f"synthetic fixture, n={len(examples)}")


def save_example_images(examples: list[dict[str, Any]], out_dir: str | Path) -> None:
    root = ensure_dir(out_dir)
    for ex in examples:
        path = root / f"{ex['example_id']}_input.png"
        Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).save(path)
        ex["input_image_path"] = str(path)


def load_natural_text_dataset(data_cfg: dict[str, Any]) -> NaturalTextBundle:
    """Dispatch loader based on ``data_cfg['mode']``.

    Modes: ``synthetic`` (default), ``local``, ``textocr``, ``coco_text``,
    ``open_images``.
    """

    mode = str(data_cfg.get("mode", "synthetic")).lower()
    image_size = data_cfg.get("image_size", 224)
    image_size = int(image_size) if image_size else None
    split = str(data_cfg.get("split", "test"))
    if mode == "synthetic":
        syn = data_cfg.get("synthetic", {})
        return make_synthetic_natural_text_dataset(
            n_images=int(syn.get("n_images", 12)),
            size=int(syn.get("size", image_size or 96)),
            seed=int(syn.get("seed", 0)),
            split=split,
        )
    if mode == "local":
        local = data_cfg.get("local", {})
        return load_local_folder_dataset(
            root=local.get("root", "data/natural_text_images"),
            metadata_csv=local.get("metadata_csv"),
            image_size=image_size,
            split=split,
        )
    if mode in {"textocr", "coco_text", "open_images"}:
        sub = data_cfg.get(mode, {})
        loader = {
            "textocr": load_textocr_metadata,
            "coco_text": load_coco_text_metadata,
            "open_images": load_open_images_metadata,
        }[mode]
        return loader(
            sub.get("metadata_json"),
            image_root=sub.get("image_root"),
            image_size=image_size,
            split=split,
        )
    raise ValueError(f"unknown natural-text data mode: {mode}")
