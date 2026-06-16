from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from causal_reliability.experiments import run_spatial_resolution_audit as mod


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_iou_identical_boxes_is_one():
    assert mod.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert mod.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # two 10x10 boxes overlapping in a 10x5 strip -> inter=50, union=150
    assert mod.iou((0, 0, 10, 10), (0, 5, 10, 15)) == pytest.approx(50.0 / 150.0)


def test_iou_handles_none():
    assert mod.iou(None, (0, 0, 1, 1)) == 0.0


def test_parse_bbox_comma_and_numpy_forms_agree():
    a = mod.parse_bbox("[73, 67, 149, 91]")
    b = mod.parse_bbox("[73\n 67\n 149\n 91]")
    assert a == b == (73.0, 67.0, 149.0, 91.0)


def test_parse_bbox_missing_returns_none():
    assert mod.parse_bbox(None) is None
    assert mod.parse_bbox(float("nan")) is None
    assert mod.parse_bbox("nan") is None
    assert mod.parse_bbox("[1, 2, 3]") is None  # wrong arity


# --------------------------------------------------------------------------- #
# Shortcut coverage
# --------------------------------------------------------------------------- #
def test_coverage_full_containment_is_one():
    # region fully contains the target -> all of target is covered
    assert mod.coverage((0, 0, 100, 100), (10, 10, 20, 20)) == pytest.approx(1.0)


def test_coverage_partial():
    # target 10x10 at (0,0); region covers its left half (5x10) -> 0.5
    assert mod.coverage((0, 0, 5, 10), (0, 0, 10, 10)) == pytest.approx(0.5)


def test_coverage_can_be_high_when_iou_is_low():
    # A large region covers a small oracle box fully (coverage=1) but IoU is low.
    region = (0, 0, 100, 100)
    oracle = (40, 40, 60, 60)
    assert mod.coverage(region, oracle) == pytest.approx(1.0)
    assert mod.iou(region, oracle) < 0.2


def test_coverage_no_target_is_zero():
    assert mod.coverage((0, 0, 10, 10), None) == 0.0


# --------------------------------------------------------------------------- #
# Area fraction
# --------------------------------------------------------------------------- #
def test_area_fraction_basic():
    # 112x112 box in a 224x224 image -> quarter area
    assert mod.area_fraction((0, 0, 112, 112), 224 * 224) == pytest.approx(0.25)


def test_area_fraction_missing_box_is_nan():
    assert math.isnan(mod.area_fraction(None, 224 * 224))


# --------------------------------------------------------------------------- #
# Bucket assignment
# --------------------------------------------------------------------------- #
BUCKETS = [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5], [0.5, 1.01]]


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "<0.1"),
        (0.05, "<0.1"),
        (0.1, "0.1-0.3"),
        (0.29, "0.1-0.3"),
        (0.3, "0.3-0.5"),
        (0.49, "0.3-0.5"),
        (0.5, ">=0.5"),
        (1.0, ">=0.5"),
    ],
)
def test_assign_iou_bucket(value, expected):
    assert mod.assign_iou_bucket(value, BUCKETS) == expected


def test_assign_iou_bucket_nan_is_none():
    assert mod.assign_iou_bucket(float("nan"), BUCKETS) is None
    assert mod.assign_iou_bucket(None, BUCKETS) is None


def test_hit_at_threshold():
    assert mod.hit_at([0.2, 0.4, 0.6], 0.5) == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# No oracle leakage in refinement selection
# --------------------------------------------------------------------------- #
def test_refinement_selection_has_no_oracle_parameters():
    forbidden = {
        "oracle_box",
        "oracle_shortcut_bbox",
        "harmful_bbox",
        "harmful_bbox_eval_only",
        "harmful_iou",
        "shortcut_box",
        "true_label",
        "label",
        "ground_truth",
        "correctness",
        "repaired_correct",
        "is_correct",
    }
    for fn in (mod.select_refined_region, mod.nonoracle_region_score, mod.generate_refinement_variants):
        params = set(inspect.signature(fn).parameters)
        assert not (params & forbidden), f"{fn.__name__} exposes oracle-leaking params: {params & forbidden}"


def test_refinement_selection_ignores_oracle_box_by_construction():
    # Selection must depend only on candidate boxes + scores + pixels. Changing
    # the (nonexistent) oracle cannot change the result because it isn't an input.
    top = (50.0, 50.0, 150.0, 150.0)
    candidate_boxes = [(48.0, 48.0, 152.0, 152.0), (10.0, 10.0, 40.0, 40.0)]
    candidate_scores = [0.9, 0.1]
    refine_cfg = {"shrink_fractions": [0.1], "shifts": ["up"], "shift_fraction": 0.1, "split_2x2": True}
    refined, variants = mod.select_refined_region(top, candidate_boxes, candidate_scores, 224, refine_cfg)
    assert refined in variants
    # The high-score candidate overlaps the top box, so consensus keeps the
    # refined region near the top region rather than the low-score far box.
    assert mod.iou(refined, candidate_boxes[0]) > mod.iou(refined, candidate_boxes[1])


# --------------------------------------------------------------------------- #
# End-to-end on a tiny synthetic benchmark
# --------------------------------------------------------------------------- #
def _write_benchmark(bench_dir: Path, with_object: bool, with_rankings: bool):
    bench_dir.mkdir(parents=True, exist_ok=True)
    oracle = "[40, 40, 60, 60]"
    rows = []

    def cert_row(ex, method, sel_bbox, harmful_iou, obj_iou, repaired_correct,
                 orig_idx=0, rep_idx=1, orig_correct=False):
        row = {
            "example_id": ex,
            "regime": "hard_multi_decoy_misleading",
            "method": method,
            "true_label": "circle",
            "original_prediction_index": orig_idx,
            "original_correct": orig_correct,
            "repaired_prediction_index": rep_idx,
            "repaired_correct": repaired_correct,
            "selected_bbox": sel_bbox,
            "selected_harmful_iou": harmful_iou,
        }
        if with_object:
            row["selected_object_iou"] = obj_iou
        return row

    # ex 0: large region fully covering oracle (low IoU, high coverage), repairs
    # ex 1: tight region overlapping oracle well (higher IoU), repairs
    specs = [
        (0, "[0, 0, 100, 100]", 0.04, 0.2, True),
        (1, "[40, 40, 65, 65]", 0.55, 0.3, True),
    ]
    for ex, bbox, hiou, oiou, ok in specs:
        for method in (
            "nonoracle_cic_top1_repair",
            "nonoracle_cic_top3_repair",
            "nonoracle_cic_clean_safe_repair",
        ):
            rows.append(cert_row(ex, method, bbox, hiou, oiou, ok))
        # oracle method: repairs correctly -> defines true label index = 1
        rows.append(cert_row(ex, "oracle_harmful_text_neutralization", oracle, 1.0, 0.0, True))
    pd.DataFrame(rows).to_csv(bench_dir / "certs.csv", index=False)

    if with_rankings:
        rk = []
        for ex, bbox, hiou, oiou, ok in specs:
            rk.append({
                "example_id": ex, "rank": 1, "bbox": bbox, "score": 0.9,
                "harmful_bbox_eval_only": oracle,
                "neutralized_prediction_index": 1, "original_prediction_index": 0,
            })
            rk.append({
                "example_id": ex, "rank": 2, "bbox": "[10, 10, 30, 30]", "score": 0.2,
                "harmful_bbox_eval_only": oracle,
                "neutralized_prediction_index": 0, "original_prediction_index": 0,
            })
        pd.DataFrame(rk).to_csv(bench_dir / "rankings.csv", index=False)


def _base_cfg(results_dir: Path, bench: dict) -> dict:
    return {
        "results_dir": str(results_dir),
        "output_subdir": "spatial_resolution_audit",
        "image_size": 224,
        "selected_method": "nonoracle_cic_top1_repair",
        "top3_method": "nonoracle_cic_top3_repair",
        "clean_safe_method": "nonoracle_cic_clean_safe_repair",
        "oracle_method": "oracle_harmful_text_neutralization",
        "shortcut_iou_column": "selected_harmful_iou",
        "object_iou_column": "selected_object_iou",
        "shortcut_regimes": ["hard_multi_decoy_misleading"],
        "iou_thresholds": [0.1, 0.2, 0.3, 0.4, 0.5],
        "iou_buckets": BUCKETS,
        "refinement": {"enabled": True, "shrink_fractions": [0.1, 0.2],
                       "shifts": ["up", "down", "left", "right"], "shift_fraction": 0.1,
                       "split_2x2": True, "area_weight": 0.25, "consensus_weight": 1.0},
        "benchmarks": [bench],
    }


def test_end_to_end_outputs_written(tmp_path: Path):
    results_dir = tmp_path / "results"
    bench_dir = results_dir / "bench_a"
    _write_benchmark(bench_dir, with_object=True, with_rankings=True)
    bench = {"name": "bench_a", "dir": "bench_a", "certificates": "certs.csv", "rankings": "rankings.csv"}
    out = mod.run(_base_cfg(results_dir, bench))

    out_dir = results_dir / "spatial_resolution_audit"
    for fname in (
        "spatial_resolution_summary.md",
        "spatial_resolution_key_numbers.json",
        "spatial_resolution_metrics.csv",
        "spatial_resolution_by_bucket.csv",
        "spatial_resolution_plot.png",
    ):
        assert (out_dir / fname).exists(), f"missing output: {fname}"

    metrics = pd.read_csv(out_dir / "spatial_resolution_metrics.csv")
    assert len(metrics) == 2
    # ex 0: low IoU but full shortcut coverage -> coverage>=0.8 flagged True
    ex0 = metrics[metrics.example_id == 0].iloc[0]
    assert ex0["iou"] < 0.1
    assert ex0["shortcut_coverage"] == pytest.approx(1.0)
    assert bool(ex0["shortcut_coverage_ge_0_8"]) is True
    # object overlap available here
    saved = json.loads((out_dir / "spatial_resolution_key_numbers.json").read_text())
    assert saved["pooled"]["object_overlap_available"] is True
    assert out["refinement"]["evaluated"] is True


def test_missing_object_bbox_reported_as_na(tmp_path: Path):
    results_dir = tmp_path / "results"
    bench_dir = results_dir / "bench_no_obj"
    _write_benchmark(bench_dir, with_object=False, with_rankings=True)
    bench = {"name": "bench_no_obj", "dir": "bench_no_obj", "certificates": "certs.csv",
             "rankings": "rankings.csv", "object_iou_column": None}
    cfg = _base_cfg(results_dir, bench)
    cfg["object_iou_column"] = None
    out = mod.run(cfg)
    assert out["pooled"]["object_overlap_available"] is False
    assert out["pooled"]["object_iou_median"] is None
    metrics = pd.read_csv(results_dir / "spatial_resolution_audit" / "spatial_resolution_metrics.csv")
    assert metrics["object_iou"].isna().all()


def test_missing_oracle_box_makes_coverage_na(tmp_path: Path):
    # No rankings file -> no oracle box on disk -> coverage / area_frac_oracle n/a,
    # but IoU (from the stored column) and repair metrics still computed.
    results_dir = tmp_path / "results"
    bench_dir = results_dir / "bench_no_oracle"
    _write_benchmark(bench_dir, with_object=True, with_rankings=False)
    bench = {"name": "bench_no_oracle", "dir": "bench_no_oracle", "certificates": "certs.csv"}
    out = mod.run(_base_cfg(results_dir, bench))
    assert out["pooled"]["shortcut_coverage_median"] is None
    assert out["pooled"]["median_iou"] is not None
    metrics = pd.read_csv(results_dir / "spatial_resolution_audit" / "spatial_resolution_metrics.csv")
    assert metrics["shortcut_coverage"].isna().all()
    assert metrics["iou"].notna().all()
