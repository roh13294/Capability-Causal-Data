from __future__ import annotations

"""Build the COCO-Text x COCO-objects metadata CSV for the natural-text CIC.

Experiment name: ``coco_text_cic_metadata``.

This runner detects the locally-present COCO 2014 *train* inputs, joins COCO
object annotations with COCO-Text v2 by ``image_id``, applies a small set of
quality filters, and writes ``data/coco_text_cic/metadata.csv`` (the schema the
``local`` adapter in ``natural_text_dataset`` reads).

It does NOT run CLIP/CIC, writes nothing under ``results/``, and therefore cannot
change any final-report metric or curated natural-text Round-1 artifact.

Usage::

    python -m causal_reliability.experiments.build_coco_text_cic_metadata \
        --max_images 500
"""

import argparse
import json
from pathlib import Path
from typing import Any

from causal_reliability.data.coco_text_cic_builder import (
    DEFAULT_ALLOWED_CATEGORIES,
    FilterConfig,
    build_metadata,
    detect_paths,
    load_captions,
)
from causal_reliability.utils.io import ensure_dir

CSV_COLUMNS = [
    "image_path",
    "human_label",
    "allowed_clip_labels",
    "optional_text_boxes",
    "optional_object_boxes",
    "source",
    "notes",
]


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    import yaml

    with p.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run(
    dataset_root: str | Path,
    cfg: dict[str, Any],
    max_images: int | None,
    write: bool = True,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    raw_root = dataset_root / "raw"
    paths = detect_paths(raw_root)

    filt = FilterConfig.from_dict(cfg.get("filters"))
    prefer_set = str(cfg.get("prefer_set", "val"))
    image_path_prefix = str(cfg.get("image_path_prefix", "raw/train2014"))

    captions = load_captions(paths.get("captions"))

    rows, stats = build_metadata(
        paths=paths,
        cfg=filt,
        max_images=max_images,
        prefer_set=prefer_set,
        image_path_prefix=image_path_prefix,
        captions=captions,
    )

    csv_path = dataset_root / "metadata.csv"
    if write:
        ensure_dir(dataset_root)
        import pandas as pd

        frame = pd.DataFrame([{c: r.get(c, "") for c in CSV_COLUMNS} for r in rows], columns=CSV_COLUMNS)
        frame.to_csv(csv_path, index=False)

    return {
        "paths_detected": {k: (str(v) if v else None) for k, v in paths.items()},
        "stats": vars(stats),
        "metadata_csv": str(csv_path),
        "rows": rows,
        "allowed_categories": list(filt.allowed_categories),
    }


def _print_report(result: dict[str, Any], max_images: int | None) -> None:
    stats = result["stats"]
    paths = result["paths_detected"]
    rows = result["rows"]

    print("=" * 72)
    print("COCO-Text x COCO-objects metadata build")
    print("=" * 72)
    print("\nDetected inputs:")
    for k, v in paths.items():
        print(f"  {k:>12}: {v}")

    print("\nCounts:")
    print(f"  COCO-Text images loaded:                 {stats['coco_text_images_loaded']}")
    print(f"    of which COCO-Text set == 'val':       {stats['coco_text_val_images']}")
    print(f"  train2014 images found on disk:          {stats['train2014_images_found']}")
    print(f"  COCO instances images loaded:            {stats['instances_images_loaded']}")
    print(f"  image IDs with BOTH object + text anns:  {stats['ids_with_both']}")
    print(f"  image IDs after filtering (rows):        {stats['ids_after_filtering']}")
    if max_images is not None:
        print(f"  (--max_images cap:                       {max_images})")

    print(f"\nMetadata CSV path: {result['metadata_csv']}")

    print("\nSample (up to 10 rows):")
    for r in rows[:10]:
        n_text = r["optional_text_boxes"].count("],") + (1 if r["optional_text_boxes"] not in ("[]", "") else 0)
        n_obj = r["optional_object_boxes"].count("],") + (1 if r["optional_object_boxes"] not in ("[]", "") else 0)
        print(f"  - {r['image_path']:<40} label={r['human_label']:<12} "
              f"text_boxes={n_text} object_boxes={n_obj}")

    issues = stats["schema_issues"]
    print("\nSchema issues:", "none" if not issues else "")
    for s in issues:
        print(f"  - {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default="data/coco_text_cic")
    parser.add_argument("--config", default="configs/coco_text_cic_metadata.yaml")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--dry_run", action="store_true", help="Do not write the CSV.")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    result = run(
        dataset_root=args.dataset_root,
        cfg=cfg,
        max_images=args.max_images,
        write=not args.dry_run,
    )
    _print_report(result, args.max_images)


if __name__ == "__main__":
    main()
