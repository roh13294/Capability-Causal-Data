from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from causal_reliability.analysis import coco_text_localization_diagnostic as lib
from causal_reliability.experiments import run_coco_text_cic_localization_diagnostic as runner


# --------------------------------------------------------------------------- #
# Proposal recall metrics
# --------------------------------------------------------------------------- #
def test_iou_and_coverage_basic():
    a = (0, 0, 10, 10)
    assert lib.iou(a, a) == pytest.approx(1.0)
    # half overlap in x: intersection 5x10=50, union 100+100-50=150
    assert lib.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    # coverage of the small target box fully contained
    assert lib.coverage_of_box((0, 0, 20, 20), (5, 5, 15, 15)) == pytest.approx(1.0)
    # proposal covers half of the target's area
    assert lib.coverage_of_box((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)


def test_proposal_recall_metrics_thresholds():
    text_boxes = [(50, 50, 70, 60)]  # area 200
    object_boxes = [(0, 0, 40, 40)]
    # one proposal exactly on the text box (IoU 1, coverage 1), one on the object,
    # one background patch overlapping neither.
    prop_boxes = [(50, 50, 70, 60), (0, 0, 40, 40), (90, 90, 100, 100)]
    m = lib.proposal_recall_metrics(prop_boxes, text_boxes, object_boxes)
    assert m["best_text_iou"] == pytest.approx(1.0)
    assert m["best_text_coverage"] == pytest.approx(1.0)
    assert m["best_object_iou"] == pytest.approx(1.0)
    assert m["has_text_iou_010"] is True
    assert m["has_text_iou_030"] is True
    assert m["has_text_iou_050"] is True
    assert m["has_text_coverage_030"] is True
    assert m["has_text_coverage_080"] is True
    assert m["n_text_overlapping_proposals"] == 1
    assert m["n_object_overlapping_proposals"] == 1


def test_proposal_recall_no_text_overlap():
    text_boxes = [(50, 50, 70, 60)]
    prop_boxes = [(0, 0, 5, 5), (90, 90, 100, 100)]
    m = lib.proposal_recall_metrics(prop_boxes, text_boxes, [])
    assert m["best_text_iou"] == 0.0
    assert m["has_text_iou_010"] is False
    assert m["n_text_overlapping_proposals"] == 0
    assert m["best_object_iou"] == 0.0


# --------------------------------------------------------------------------- #
# text-overlap@k and rank of best text proposal
# --------------------------------------------------------------------------- #
def test_text_overlap_at_k():
    # text overlap first appears at rank 4
    flags = [False, False, False, True, False, True]
    out = lib.text_overlap_at_k(flags, ks=(1, 3, 5, 10))
    assert out[1] is False
    assert out[3] is False
    assert out[5] is True
    assert out[10] is True


def test_rank_of_first_true():
    assert lib.rank_of_first_true([False, False, True, False]) == 3
    assert lib.rank_of_first_true([True, True]) == 1
    assert lib.rank_of_first_true([False, False]) is None


# --------------------------------------------------------------------------- #
# Area-normalized score ordering
# --------------------------------------------------------------------------- #
def test_adjusted_score_modes():
    # smaller area gets boosted by div_sqrt_area / div_area_clip
    assert lib.adjusted_score(1.0, 0.25, mode="div_sqrt_area") == pytest.approx(2.0)
    assert lib.adjusted_score(1.0, 0.01, mode="div_area_clip", area_floor=0.02) == pytest.approx(50.0)
    # object overlap penalised
    assert lib.adjusted_score(1.0, 0.1, mode="penalize_object", overlaps_object=True, object_penalty=0.8) == pytest.approx(0.2)
    assert lib.adjusted_score(1.0, 0.1, mode="penalize_object", overlaps_object=False) == pytest.approx(1.0)
    # oracle text reward
    assert lib.adjusted_score(1.0, 0.1, mode="reward_text_oracle", text_iou=0.5, text_reward=2.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        lib.adjusted_score(1.0, 0.1, mode="nonsense")


def test_reorder_changes_top1_for_small_text_region():
    # A large object region scores highest originally; a small text region is rank 2.
    items = [
        lib.ScoredProposal("obj", score=1.0, area_fraction=0.40, overlaps_text=False, overlaps_object=True, text_iou=0.0),
        lib.ScoredProposal("txt", score=0.8, area_fraction=0.02, overlaps_text=True, overlaps_object=False, text_iou=0.6),
    ]
    assert lib.reorder(items, "original")[0].candidate_id == "obj"
    # area normalization should lift the small text region to the top
    assert lib.reorder(items, "div_sqrt_area")[0].candidate_id == "txt"
    # object penalty should also lift the text region
    assert lib.reorder(items, "penalize_object")[0].candidate_id == "txt"
    # oracle text reward must lift the text region
    assert lib.reorder(items, "reward_text_oracle")[0].candidate_id == "txt"


def test_reorder_stable_on_ties():
    items = [
        lib.ScoredProposal("a", score=1.0, area_fraction=0.1, overlaps_text=False, overlaps_object=False),
        lib.ScoredProposal("b", score=1.0, area_fraction=0.1, overlaps_text=False, overlaps_object=False),
    ]
    assert [it.candidate_id for it in lib.reorder(items, "original")] == ["a", "b"]


def test_leakage_mode_flagged():
    assert lib.is_leakage_mode("reward_text_oracle") is True
    assert lib.is_leakage_mode("div_sqrt_area") is False


# --------------------------------------------------------------------------- #
# Top-k union construction
# --------------------------------------------------------------------------- #
def test_topk_union_ids():
    ranked = ["r1", "r2", "r3", "r4", "r5"]
    assert lib.topk_union_ids(ranked, 1) == ["r1"]
    assert lib.topk_union_ids(ranked, 3) == ["r1", "r2", "r3"]
    assert lib.topk_union_ids(ranked, 10) == ranked


def test_topk_union_dedup():
    assert lib.topk_union_ids(["a", "a", "b", "c"], 3) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Recall + ranking integration over a tiny synthetic example
# --------------------------------------------------------------------------- #
def test_compute_recall_and_ranking_ranking_failure_shape():
    import json

    # Two open proposals: rank-1 background (no text), rank-2 text-overlapping.
    diag = pd.DataFrame([
        {"example_id": 0, "rank": 1, "candidate_id": "bg", "proposal_type": "grid_patch_18",
         "proposal_family": "grid_patch", "bbox": json.dumps([0, 0, 40, 40]), "score": 1.0,
         "area_fraction": 0.3, "overlaps_text_box": False, "overlaps_object_box": True},
        {"example_id": 0, "rank": 2, "candidate_id": "txt", "proposal_type": "textness_high_frequency",
         "proposal_family": "connected_component", "bbox": json.dumps([50, 50, 70, 60]), "score": 0.8,
         "area_fraction": 0.02, "overlaps_text_box": True, "overlaps_object_box": False},
        # an OCR-family proposal that must be excluded from the open pool
        {"example_id": 0, "rank": 3, "candidate_id": "ocr", "proposal_type": "ocr_text_box",
         "proposal_family": "ocr_text_box", "bbox": json.dumps([50, 50, 70, 60]), "score": 0.9,
         "area_fraction": 0.02, "overlaps_text_box": True, "overlaps_object_box": False},
    ])
    diag_by_example = {0: diag}
    boxes_by_example = {0: {"text": [(50, 50, 70, 60)], "object": [(0, 0, 40, 40)]}}
    df = runner.compute_recall_and_ranking(diag_by_example, boxes_by_example)
    row = df.iloc[0]
    # text-overlapping proposal exists (recall) but is not ranked first (ranking issue)
    assert bool(row["has_text_overlapping_proposal"]) is True
    assert row["rank_best_text_proposal"] == 2
    assert bool(row["selected_overlaps_text"]) is False
    assert bool(row["text_overlap_at_1"]) is False
    assert bool(row["text_overlap_at_3"]) is True
    # OCR family excluded: only 1 text-overlapping proposal counted in the open pool
    assert row["n_text_overlapping_proposals"] == 1


# --------------------------------------------------------------------------- #
# Guardrail: protected paths and no final-metrics modification
# --------------------------------------------------------------------------- #
def test_refuses_to_write_protected_paths():
    with pytest.raises(RuntimeError):
        runner._assert_safe_output(Path("results/coco_text_cic_full"))
    with pytest.raises(RuntimeError):
        runner._assert_safe_output(Path("results/final_report"))
    with pytest.raises(RuntimeError):
        runner._assert_safe_output(Path("results/final_report/subdir"))
    # the intended output dir is allowed
    runner._assert_safe_output(Path("results/coco_text_cic_localization_diagnostic"))


def test_final_report_files_untouched_by_import():
    # Importing / running the pure helpers must never mutate final-report artifacts.
    fr = Path("results/final_report/final_key_numbers.json")
    if not fr.exists():
        pytest.skip("final_report not present in this checkout")
    before = hashlib.sha256(fr.read_bytes()).hexdigest()
    # exercise pure helpers
    lib.proposal_recall_metrics([(0, 0, 10, 10)], [(0, 0, 10, 10)], [])
    lib.reorder([lib.ScoredProposal("x", 1.0, 0.1, False, False)], "div_sqrt_area")
    after = hashlib.sha256(fr.read_bytes()).hexdigest()
    assert before == after
