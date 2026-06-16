"""Tests for the lightweight CIC demo + external validation scaffold.

These tests exercise the demo in mock mode only (no model downloads, no server
launch) and assert the demo is non-invasive: it does not change experimental
metrics, support gates, or results/final_report/, and ships no raw datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo"
SAMPLES = DEMO / "sample_images"
VALIDATION = REPO / "results" / "demo_validation"
FINAL_REPORT = REPO / "results" / "final_report"

SAMPLE_NAMES = [
    "text_overlay_success",
    "semantic_decoy_success",
    "coco_text_directional",
    "failure_abstain",
]


# --------------------------------------------------------------------------- #
# imports / config / assets
# --------------------------------------------------------------------------- #
def test_demo_modules_import_without_launching_server():
    # Importing the app must not require gradio and must not start a server.
    import demo.app as app
    import demo.cic_pipeline as pipeline

    assert hasattr(pipeline, "run_pipeline")
    assert hasattr(app, "build_interface")  # not called -> no server


def test_demo_config_loads():
    from demo.cic_pipeline import DemoConfig

    cfg = DemoConfig.from_yaml(DEMO / "demo_config.yaml")
    assert cfg.mode in ("mock", "real", "auto")
    assert cfg.class_names and isinstance(cfg.class_names, list)
    assert cfg.top_k >= 1
    assert cfg.max_candidates >= 1


def test_app_load_config_returns_config():
    from demo.app import load_config

    cfg = load_config()
    assert cfg.mode in ("mock", "real", "auto")


def test_sample_image_paths_resolve():
    manifest = json.loads((SAMPLES / "manifest.json").read_text())
    files = {item["file"] for item in manifest["images"]}
    for name in SAMPLE_NAMES:
        path = SAMPLES / f"{name}.png"
        assert path.exists(), f"missing sample {path}"
        assert f"{name}.png" in files
        Image.open(path).verify()  # valid image


def test_app_lists_sample_images():
    from demo.app import list_sample_images, load_config

    samples = list_sample_images(load_config())
    assert len(samples) >= 4
    assert all(s.endswith(".png") for s in samples)


# --------------------------------------------------------------------------- #
# mock pipeline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mock_config():
    from demo.cic_pipeline import DemoConfig

    return DemoConfig(mode="mock")


def _run(name, cfg):
    from demo.cic_pipeline import run_pipeline

    return run_pipeline(Image.open(SAMPLES / f"{name}.png"), cfg)


def test_mock_pipeline_runs_on_all_samples(mock_config):
    for name in SAMPLE_NAMES:
        result = _run(name, mock_config)
        assert result.mode_used == "mock"
        assert result.backend == "deterministic_stub"
        assert len(result.original_top_k) >= 1
        assert len(result.repaired_top_k) >= 1
        assert 0.0 <= result.original_top_k[0]["confidence"] <= 1.0
        assert result.reliability_action in ("repair", "abstain")
        assert result.n_candidates >= 1
        assert result.scope_note  # scope note always present


def test_mock_pipeline_demonstrates_repair_and_abstain(mock_config):
    # The demo's pedagogical value: at least one clear shortcut repair and at
    # least one abstain across the curated samples.
    results = {name: _run(name, mock_config) for name in SAMPLE_NAMES}
    assert any(r.prediction_changed for r in results.values()), "expected >=1 repair flip"
    assert any(r.reliability_action == "abstain" for r in results.values()), "expected >=1 abstain"
    # failure_abstain is curated to abstain.
    assert results["failure_abstain"].reliability_action == "abstain"


def test_resolve_predict_fn_mock(mock_config):
    from demo.cic_pipeline import resolve_predict_fn

    fn, mode_used, backend = resolve_predict_fn(mock_config)
    assert mode_used == "mock"
    assert backend == "deterministic_stub"
    probs = fn([Image.open(SAMPLES / "text_overlay_success.png")])
    assert probs.shape == (1, len(mock_config.class_names))
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_format_summary_includes_scope_note(mock_config):
    from demo.app import format_summary
    from demo.cic_pipeline import SCOPE_NOTE

    summary = format_summary(_run("text_overlay_success", mock_config))
    assert SCOPE_NOTE in summary
    assert "Original top-5" in summary


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_report_writes_json_and_png(tmp_path, mock_config):
    from demo.cic_pipeline import export_report

    img = Image.open(SAMPLES / "text_overlay_success.png")
    result = _run("text_overlay_success", mock_config)
    paths = export_report(img, result, tmp_path, name="unit_report")

    json_path = Path(paths["json"])
    png_path = Path(paths["png"])
    assert json_path.exists() and png_path.exists()
    # written only inside the requested dir
    assert tmp_path in json_path.parents and tmp_path in png_path.parents

    payload = json.loads(json_path.read_text())
    assert payload["scope_note"]
    assert "original_top_k" in payload and "repaired_top_k" in payload
    assert payload["mode_used"] == "mock"


# --------------------------------------------------------------------------- #
# external validation scaffold present
# --------------------------------------------------------------------------- #
def test_external_validation_scaffold_exists():
    for fname in (
        "external_validation_protocol.md",
        "external_validation_form.md",
        "external_validation_template.csv",
    ):
        path = VALIDATION / fname
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 0


def test_validation_protocol_uses_bounded_language():
    text = (VALIDATION / "external_validation_protocol.md").read_text().lower()
    assert "external demonstration feedback" in text
    assert "small usability check" in text
    # forbidden over-claims must be explicitly disallowed, never asserted as done.
    for forbidden in ("deployment validation", "clinical validation", "proof of real-world robustness"):
        # appears only in the "must NOT claim" section, prefixed by a negation marker.
        assert forbidden in text  # documented as disallowed
        assert f"❌ \"{forbidden}\"" in (VALIDATION / "external_validation_protocol.md").read_text().lower()


# --------------------------------------------------------------------------- #
# non-invasiveness guards
# --------------------------------------------------------------------------- #
def _snapshot(directory: Path):
    return {
        p.relative_to(directory).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_final_report_exists_and_untouched_by_demo(tmp_path, mock_config):
    assert FINAL_REPORT.exists(), "results/final_report/ must exist"
    before = _snapshot(FINAL_REPORT)
    assert before, "final_report should contain artifacts"

    # Run the full demo flow incl. export (to tmp, never to final_report).
    from demo.cic_pipeline import export_report

    for name in SAMPLE_NAMES:
        result = _run(name, mock_config)
        export_report(Image.open(SAMPLES / f"{name}.png"), result, tmp_path, name=f"r_{name}")

    after = _snapshot(FINAL_REPORT)
    assert before == after, "demo must not modify results/final_report/"


def test_support_gate_modules_unchanged_and_importable():
    # Support / abstention gate logic must remain importable and is not altered
    # by the demo (the demo imports, never writes, these modules).
    from causal_reliability.repair import abstention
    from causal_reliability.analysis import predictive_cic_gate

    assert hasattr(abstention, "selective_abstention_policy")
    assert hasattr(abstention, "select_abstention_threshold")
    assert hasattr(predictive_cic_gate, "assert_label_free")


def test_demo_ships_no_raw_datasets():
    # No large blobs, archives, or raw dataset folders inside demo/.
    forbidden_names = {"train2014", "val2014", "coco", "coco_text", "wilds"}
    for path in DEMO.rglob("*"):
        if path.is_dir():
            assert path.name.lower() not in forbidden_names, f"raw dataset dir in demo: {path}"
        if path.is_file():
            assert path.suffix.lower() not in {".zip", ".tar", ".gz", ".npz"}, f"archive in demo: {path}"
            # sample/static assets should stay small (synthetic, not real datasets).
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                assert path.stat().st_size < 512 * 1024, f"oversized image in demo: {path}"
