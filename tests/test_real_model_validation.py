from pathlib import Path
import os
import sys
from types import SimpleNamespace

from causal_reliability.analysis.final_report import build_report
from causal_reliability.analysis.metrics import auroc_with_reason
from causal_reliability.analysis.main_table import build_main_table
from causal_reliability.data.clip_overlay_shortcuts import make_clip_overlay_dataset
from causal_reliability.data.real_model_shortcuts import make_real_model_shortcut_dataset
from causal_reliability.experiments.run_clip_overlay_validation import run as run_clip_overlay_validation
from causal_reliability.experiments.run_real_model_validation import run as run_real_model_validation
from causal_reliability.real_models.clip_zero_shot import ClipStatus, check_clip_available, load_clip_model


def test_real_model_shortcuts_counterfactual_preserves_label():
    bundle = make_real_model_shortcut_dataset(n_per_class=2, size=32, shortcut_type="mixed", seed=3)
    assert len(bundle.id_examples) == len(bundle.shifted_examples)
    first = bundle.shifted_examples[0]
    assert first["label"] == 0
    assert first["image"].shape == first["counterfactual_image"].shape
    assert first["shortcut_label"] != bundle.id_examples[0]["shortcut_label"]


def test_clip_loader_fails_gracefully_if_unavailable():
    status = load_clip_model(device="cpu")
    assert hasattr(status, "available")
    assert hasattr(status, "backend")
    assert hasattr(status, "model_name")
    assert hasattr(status, "pretrained")
    if not status.available:
        assert status.message


def test_auroc_undefined_when_all_shifted_examples_fail():
    auc, reason = auroc_with_reason([0.2, 0.8, 0.9], [1, 1, 1], min_failures=1)
    assert auc != auc
    assert reason == "AUROC undefined because shifted correctness contains only one class."


def test_clip_availability_check_returns_structured_status():
    status = check_clip_available(device="cpu")
    assert isinstance(status.available, bool)
    assert isinstance(status.backend, str)
    assert isinstance(status.pretrained, bool)
    if not status.available:
        assert status.error_message


def test_clip_loader_does_not_download_when_disabled(monkeypatch):
    called = {}

    def fail_if_called(*args, **kwargs):
        called["hf_offline"] = kwargs["pretrained"] == "remote-tag" and os.environ.get("HF_HUB_OFFLINE") == "1"
        raise RuntimeError("cache miss")

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        SimpleNamespace(create_model_and_transforms=fail_if_called, get_tokenizer=lambda model_name: object()),
    )
    status = check_clip_available(device="cpu", allow_download=False, preferred_backend="open_clip", model_name="ViT-B-32", pretrained_tag="remote-tag")

    assert called["hf_offline"] is True
    assert status.available is False
    assert status.downloads_allowed is False
    assert "allow_pretrained_download true" in status.error_message


def test_clip_loader_attempts_open_clip_when_download_allowed(monkeypatch):
    called = {}

    class FakeModel:
        def eval(self):
            called["eval"] = True

    def create_model_and_transforms(model_name, pretrained, device):
        called.update({"model_name": model_name, "pretrained": pretrained, "device": device})
        return FakeModel(), None, object()

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        SimpleNamespace(create_model_and_transforms=create_model_and_transforms, get_tokenizer=lambda model_name: object()),
    )
    status = check_clip_available(
        device="cpu",
        allow_download=True,
        preferred_backend="open_clip",
        model_name="ViT-B-32",
        pretrained_tag="laion2b_s34b_b79k",
    )

    assert status.available is True
    assert status.backend == "open_clip"
    assert status.pretrained is True
    assert status.downloaded_or_cached == "download_allowed"
    assert called == {"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "device": "cpu", "eval": True}


def test_clip_cli_parses_allow_download(monkeypatch, capsys):
    from causal_reliability.real_models import clip_zero_shot as mod

    captured = {}

    def fake_check_clip_available(**kwargs):
        captured.update(kwargs)
        return ClipStatus(
            available=True,
            backend="open_clip",
            model_name=kwargs["model_name"],
            pretrained_tag=kwargs["pretrained_tag"],
            pretrained=True,
            downloaded_or_cached="download_allowed",
            device="cpu",
            downloads_allowed=kwargs["allow_download"],
            backend_attempted=kwargs["preferred_backend"],
        )

    monkeypatch.setattr(mod, "check_clip_available", fake_check_clip_available)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clip_zero_shot.py",
            "--check",
            "--allow-download",
            "--backend",
            "open_clip",
            "--model-name",
            "ViT-B-32",
            "--pretrained-tag",
            "laion2b_s34b_b79k",
        ],
    )

    mod.main()
    out = capsys.readouterr().out
    assert captured["allow_download"] is True
    assert "CLIP available: yes" in out
    assert "Pretrained weights loaded: yes" in out


def test_clip_overlay_dataset_regimes_and_counterfactual_labels():
    bundle = make_clip_overlay_dataset(n_per_class=1, size=64)
    regimes = {ex["regime"] for ex in bundle.examples}
    assert {"aligned_overlay", "misleading_overlay", "mixed_overlay"}.issubset(regimes)
    aligned = [ex for ex in bundle.examples if ex["regime"] == "aligned_overlay"]
    misleading = [ex for ex in bundle.examples if ex["regime"] == "misleading_overlay"]
    assert all(ex["shortcut_label"] == ex["label"] for ex in aligned)
    assert all(ex["shortcut_label"] != ex["label"] for ex in misleading)
    first = misleading[0]
    assert first["label"] == first["label"]
    assert first["counterfactual_image"].shape == first["image"].shape


def test_clip_overlay_runner_unavailable_writes_summary(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_clip_overlay_validation as mod
    from causal_reliability.real_models.clip_zero_shot import ClipStatus

    monkeypatch.setattr(
        mod,
        "check_clip_available",
        lambda **kwargs: ClipStatus(
            available=False,
            backend="unavailable",
            model_name="ViT-B-32",
            pretrained=False,
            downloads_allowed=kwargs.get("allow_download", False),
            backend_attempted=kwargs.get("preferred_backend", ""),
            error_message="forced unavailable",
        ),
    )
    outputs = run_clip_overlay_validation(
        {
            "results_dir": str(tmp_path),
            "data": {"n_per_class": 1, "image_size": 64},
            "model": {"allow_pretrained_download": True, "preferred_backend": "open_clip"},
        }
    )
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")
    metrics = (tmp_path / "clip_overlay_validation" / "clip_overlay_metrics.csv").read_text(encoding="utf-8")
    assert "CLIP unavailable" in summary
    assert "Downloads allowed: True" in summary
    assert "downloads_allowed" in metrics
    assert "must not be used as headline real-model validation" in summary


def test_clip_overlay_runner_passes_download_config_to_loader(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_clip_overlay_validation as mod

    captured = {}

    def fake_check_clip_available(**kwargs):
        captured.update(kwargs)
        return ClipStatus(
            available=False,
            backend="unavailable",
            model_name=kwargs["model_name"],
            pretrained=False,
            downloads_allowed=kwargs["allow_download"],
            backend_attempted=kwargs["preferred_backend"],
            error_message="forced unavailable",
        )

    monkeypatch.setattr(mod, "check_clip_available", fake_check_clip_available)
    run_clip_overlay_validation(
        {
            "results_dir": str(tmp_path),
            "data": {"n_per_class": 1, "image_size": 64},
            "model": {
                "preferred_backend": "transformers",
                "model_name": "openai/clip-vit-base-patch32",
                "allow_pretrained_download": True,
                "device": "cpu",
            },
        }
    )

    assert captured["allow_download"] is True
    assert captured["preferred_backend"] == "transformers"
    assert captured["model_name"] == "openai/clip-vit-base-patch32"


def test_clip_overlay_runner_fake_backend_writes_metrics_and_occlusion(tmp_path: Path):
    outputs = run_clip_overlay_validation(
        {
            "results_dir": str(tmp_path),
            "data": {"n_per_class": 1, "image_size": 64},
            "model": {"backend": "fake"},
        }
    )
    out = tmp_path / "clip_overlay_validation"
    assert outputs["metrics"] == str(out / "clip_overlay_metrics.csv")
    assert (out / "clip_overlay_certificates.csv").exists()
    metrics = (out / "clip_overlay_metrics.csv").read_text(encoding="utf-8")
    assert "CIC" in metrics
    summary = (out / "clip_overlay_summary.md").read_text(encoding="utf-8")
    assert "fallback smoke test" in summary
    assert "pretrained CLIP evidence" not in summary
    occ = (out / "attribution" / "clip_overlay_occlusion_metrics.csv").read_text(encoding="utf-8")
    assert "text_occlusion_drop" in occ
    assert "object_occlusion_drop" in occ
    assert "background_occlusion_drop" in occ


def test_real_model_validation_runner_smoke_fallback(tmp_path: Path):
    cfg = {
        "seed": 5,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "data": {"n_per_class": 2, "image_size": 32, "shortcut_type": "background", "classes": ["circle", "square", "triangle"]},
        "model": {"mode": "fallback", "fallback_epochs": 1, "fallback_lr": 0.01, "batch_size": 8},
    }
    outputs = run_real_model_validation(cfg)
    out = tmp_path / "real_model_validation"
    assert outputs["summary"] == str(out / "real_model_summary.md")
    assert (out / "real_model_metrics.csv").exists()
    assert (out / "real_model_certificates.csv").exists()
    assert (out / "attribution" / "attribution_metrics.csv").exists()
    summary = (out / "attribution" / "attribution_summary.md").read_text(encoding="utf-8")
    assert "sanity check" in summary
    assert "proof of mechanism" in summary


def test_final_report_includes_real_model_section(tmp_path: Path):
    cfg = {
        "seed": 7,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "data": {"n_per_class": 1, "image_size": 32, "shortcut_type": "border", "classes": ["circle", "square"]},
        "model": {"mode": "fallback", "fallback_epochs": 1, "batch_size": 4},
    }
    run_real_model_validation(cfg)
    paths = build_report(tmp_path)
    report = paths["report"].read_text(encoding="utf-8")
    assert "## Real-Model Validation" in report
    assert "not proof that CIC generalizes to all foundation models" in report
    assert "fallback smoke test" in report


def test_final_report_and_main_table_do_not_treat_fallback_as_pretrained(tmp_path: Path):
    cfg = {
        "seed": 8,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "data": {"n_per_class": 1, "image_size": 32, "shortcut_type": "border", "classes": ["circle", "square"]},
        "model": {"mode": "fallback", "fallback_epochs": 1, "batch_size": 4},
    }
    run_real_model_validation(cfg)
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    table = build_main_table(tmp_path)
    assert "fallback smoke test" in report
    real_row = table[table["Task"] == "real_model_validation"].iloc[0]
    assert real_row["Evidence Status"] == "fallback smoke test"


def test_readme_does_not_overclaim_foundation_models():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "does not prove that CIC generalizes to all foundation models" in text
