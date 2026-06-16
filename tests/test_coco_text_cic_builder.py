from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from causal_reliability.data.coco_text_cic_builder import (
    DEFAULT_ALLOWED_CATEGORIES,
    FilterConfig,
    build_metadata,
    detect_paths,
    evaluate_image,
    overlap_fraction_of_first,
    xywh_to_xyxy,
)
import causal_reliability.experiments.build_coco_text_cic_metadata as runner


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
def test_xywh_to_xyxy():
    assert xywh_to_xyxy([10, 20, 30, 40]) == (10, 20, 40, 60)


def test_overlap_fraction_of_first():
    a = (0, 0, 10, 10)  # area 100
    assert overlap_fraction_of_first(a, (0, 0, 5, 10)) == 0.5
    assert overlap_fraction_of_first(a, (100, 100, 110, 110)) == 0.0
    assert overlap_fraction_of_first((0, 0, 0, 0), a) == 0.0


# --------------------------------------------------------------------------- #
# Per-image filtering logic
# --------------------------------------------------------------------------- #
def _coco_image(width=100, height=100, iid=7):
    return {"id": iid, "width": width, "height": height, "file_name": f"COCO_train2014_{iid:012d}.jpg"}


def _obj(cat_id, box_xywh, iscrowd=0):
    return {"category_id": cat_id, "bbox": box_xywh, "iscrowd": iscrowd}


def _text(box_xywh, legibility="legible", language="english"):
    return {"bbox": box_xywh, "legibility": legibility, "language": language}


CATS = {1: "dog", 2: "person", 3: "cup"}


def test_evaluate_image_happy_path():
    cfg = FilterConfig(allowed_categories=("dog", "cup"))
    # dog box 40x40 (area frac 0.16); text box 20x10 in the corner, separable.
    objs = [_obj(1, [5, 5, 40, 40])]
    text = [_text([70, 70, 20, 10])]
    row = evaluate_image(7, _coco_image(), objs, text, CATS, cfg)
    assert row is not None
    assert row["human_label"] == "dog"
    assert len(row["text_boxes"]) == 1
    assert row["object_boxes"][0] == (5, 5, 45, 45)


def test_evaluate_image_rejects_no_allowed_object():
    cfg = FilterConfig(allowed_categories=("dog",))
    objs = [_obj(2, [5, 5, 40, 40])]  # person, not allowed
    text = [_text([70, 70, 20, 10])]
    assert evaluate_image(7, _coco_image(), objs, text, CATS, cfg) is None


def test_evaluate_image_rejects_small_object():
    cfg = FilterConfig(allowed_categories=("dog",), min_object_area_frac=0.5)
    objs = [_obj(1, [5, 5, 40, 40])]  # frac 0.16 < 0.5
    text = [_text([70, 70, 20, 10])]
    assert evaluate_image(7, _coco_image(), objs, text, CATS, cfg) is None


def test_evaluate_image_rejects_illegible_or_non_english():
    cfg = FilterConfig(allowed_categories=("dog",))
    objs = [_obj(1, [5, 5, 40, 40])]
    assert evaluate_image(7, _coco_image(), objs, [_text([70, 70, 20, 10], legibility="illegible")], CATS, cfg) is None
    assert evaluate_image(7, _coco_image(), objs, [_text([70, 70, 20, 10], language="not english")], CATS, cfg) is None


def test_evaluate_image_rejects_text_overlapping_object():
    cfg = FilterConfig(allowed_categories=("dog",), max_text_object_overlap=0.5)
    objs = [_obj(1, [0, 0, 80, 80])]  # large dog covering most of the image
    text = [_text([10, 10, 20, 20])]  # fully inside the dog box -> overlap 1.0
    assert evaluate_image(7, _coco_image(), objs, text, CATS, cfg) is None


def test_evaluate_image_rejects_too_many_dominant_objects():
    cfg = FilterConfig(allowed_categories=("dog", "cup"), dominant_object_area_frac=0.10, max_dominant_objects=1)
    objs = [_obj(1, [0, 0, 40, 40]), _obj(3, [50, 50, 40, 40])]  # two dominant objects
    text = [_text([5, 90, 20, 8])]
    assert evaluate_image(7, _coco_image(), objs, text, CATS, cfg) is None


# --------------------------------------------------------------------------- #
# End-to-end build on a tiny synthetic fixture (no network, no real COCO files)
# --------------------------------------------------------------------------- #
def _write_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "coco_text_cic"
    raw = root / "raw"
    train = raw / "train2014"
    ann = raw / "annotations"
    train.mkdir(parents=True)
    ann.mkdir(parents=True)

    fname = "COCO_train2014_000000000042.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(train / fname)

    instances = {
        "images": [{"id": 42, "width": 100, "height": 100, "file_name": fname}],
        "categories": [{"id": 1, "name": "dog", "supercategory": "animal"}],
        "annotations": [{"image_id": 42, "category_id": 1, "bbox": [5, 5, 40, 40], "iscrowd": 0, "id": 1}],
    }
    (ann / "instances_train2014.json").write_text(json.dumps(instances))
    (ann / "captions_train2014.json").write_text(
        json.dumps({"annotations": [{"image_id": 42, "caption": "a dog near a sign"}]})
    )
    cocotext = {
        "imgs": {"42": {"id": 42, "set": "val", "width": 100, "height": 100, "file_name": fname}},
        "anns": {"900": {"image_id": 42, "bbox": [70, 70, 20, 12], "legibility": "legible", "language": "english"}},
        "imgToAnns": {"42": [900]},
    }
    (raw / "cocotext.v2.json").write_text(json.dumps(cocotext))
    return root


def test_detect_paths_on_fixture(tmp_path):
    root = _write_fixture(tmp_path)
    paths = detect_paths(root / "raw")
    assert paths["train_dir"] is not None
    assert paths["instances"] is not None
    assert paths["captions"] is not None
    assert paths["cocotext"] is not None


def test_build_metadata_end_to_end(tmp_path):
    root = _write_fixture(tmp_path)
    result = runner.run(dataset_root=root, cfg={"filters": {"allowed_categories": ["dog"]}}, max_images=500)
    stats = result["stats"]
    assert stats["coco_text_images_loaded"] == 1
    assert stats["coco_text_val_images"] == 1
    assert stats["train2014_images_found"] == 1
    assert stats["ids_with_both"] == 1
    assert stats["ids_after_filtering"] == 1
    assert stats["schema_issues"] == []

    csv_path = Path(result["metadata_csv"])
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == runner.CSV_COLUMNS
    assert df.iloc[0]["human_label"] == "dog"
    assert df.iloc[0]["image_path"] == "raw/train2014/COCO_train2014_000000000042.jpg"
    assert df.iloc[0]["source"] == "coco_text_cic"
    assert "coco_text_set=val" in df.iloc[0]["notes"]


def test_max_images_cap(tmp_path):
    root = _write_fixture(tmp_path)
    result = runner.run(dataset_root=root, cfg={"filters": {"allowed_categories": ["dog"]}}, max_images=0)
    assert result["stats"]["ids_after_filtering"] == 0


def test_default_categories_exclude_person():
    assert "person" not in DEFAULT_ALLOWED_CATEGORIES
