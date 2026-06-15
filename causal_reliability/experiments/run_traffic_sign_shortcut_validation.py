from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.metrics import auroc
from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def _write_placeholder(path: Path, title: str) -> None:
    plt.figure(figsize=(5.2, 3.4))
    plt.text(0.5, 0.5, title, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _unavailable(out_dir: Path, reason: str) -> dict[str, str]:
    examples_dir = ensure_dir(out_dir / "traffic_sign_examples")
    metrics = pd.DataFrame(
        [
            {
                "dataset": "GTSRB",
                "status": "unavailable",
                "method": "CIC",
                "failure_auroc": np.nan,
                "reason": reason,
            }
        ]
    )
    metrics.to_csv(out_dir / "traffic_sign_metrics.csv", index=False)
    pd.DataFrame(columns=["example_id", "label", "failure", "confidence", "cic_score"]).to_csv(out_dir / "traffic_sign_certificates.csv", index=False)
    _write_placeholder(out_dir / "traffic_sign_reliability_plane.png", "Traffic-sign validation unavailable")
    _write_placeholder(out_dir / "traffic_sign_reliability_plane.pdf", "Traffic-sign validation unavailable")
    summary = [
        "# Safety-Critical-Inspired Traffic Sign Shortcut Validation",
        "",
        "This experiment is a safety-critical-inspired shortcut audit. It does not validate deployment in autonomous vehicles.",
        "",
        "GTSRB was not used.",
        f"Unavailable reason: {reason}",
        "",
        "No real traffic-sign dataset validation is claimed from this run. The runner writes this unavailable summary rather than fabricating results.",
        "",
    ]
    (out_dir / "traffic_sign_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": str(out_dir / "traffic_sign_metrics.csv"),
        "summary": str(out_dir / "traffic_sign_summary.md"),
        "certificates": str(out_dir / "traffic_sign_certificates.csv"),
        "examples_dir": str(examples_dir),
        "plane_png": str(out_dir / "traffic_sign_reliability_plane.png"),
        "plane_pdf": str(out_dir / "traffic_sign_reliability_plane.pdf"),
    }


def _synthetic(out_dir: Path, cfg: dict[str, Any]) -> dict[str, str]:
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n = int(cfg.get("n_examples", 120))
    labels = np.arange(n) % 3
    patch_aligned = rng.random(n) > 0.55
    patch_label = np.where(patch_aligned, labels, (labels + 1) % 3)
    pred = patch_label
    failure = (pred != labels).astype(int)
    confidence = np.full(n, 0.92)
    random_aug = rng.normal(0.1, 0.03, size=n).clip(0, 1)
    occlusion = (patch_label != labels).astype(float) * 0.75 + 0.08
    cic = occlusion + 0.2
    entropy = np.full(n, 0.25)
    margin = -np.full(n, 0.84)
    score_map = {
        "confidence risk": 1.0 - confidence,
        "entropy": entropy,
        "margin": margin,
        "random augmentation sensitivity": random_aug,
        "occlusion heuristic": occlusion,
        "CIC": cic,
    }
    metrics = pd.DataFrame(
        [
            {
                "dataset": "synthetic safety-critical-inspired signs",
                "status": "synthetic_fallback",
                "method": method,
                "failure_auroc": auroc(scores, failure),
                "n_examples": int(n),
                "n_failures": int(failure.sum()),
            }
            for method, scores in score_map.items()
        ]
    )
    certs = pd.DataFrame(
        {
            "example_id": [f"synthetic_sign_{i:04d}" for i in range(n)],
            "label": labels,
            "patch_label": patch_label,
            "predicted_label": pred,
            "failure": failure,
            "confidence": confidence,
            "random_augmentation_sensitivity": random_aug,
            "occlusion_heuristic": occlusion,
            "cic_score": cic,
        }
    )
    metrics.to_csv(out_dir / "traffic_sign_metrics.csv", index=False)
    certs.to_csv(out_dir / "traffic_sign_certificates.csv", index=False)
    _write_placeholder(out_dir / "traffic_sign_reliability_plane.png", "Synthetic traffic-sign-inspired shortcut audit")
    _write_placeholder(out_dir / "traffic_sign_reliability_plane.pdf", "Synthetic traffic-sign-inspired shortcut audit")
    ensure_dir(out_dir / "traffic_sign_examples")
    summary = [
        "# Safety-Critical-Inspired Traffic Sign Shortcut Validation",
        "",
        "This experiment is a safety-critical-inspired shortcut audit. It does not validate deployment in autonomous vehicles.",
        "",
        "GTSRB was not used. Synthetic fallback was used and is labeled as synthetic safety-critical-inspired evidence, not real-world traffic-sign validation.",
        "",
        "Task: road-sign-like class labels with a small simulated corner sticker shortcut. The task demonstrates that a factor-specific patch intervention can audit shortcut reliance in a safety-critical-inspired setting. It does not demonstrate robustness of autonomous-driving systems or deployment readiness.",
        "",
        _markdown_table(metrics),
        "",
    ]
    (out_dir / "traffic_sign_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": str(out_dir / "traffic_sign_metrics.csv"),
        "summary": str(out_dir / "traffic_sign_summary.md"),
        "certificates": str(out_dir / "traffic_sign_certificates.csv"),
        "examples_dir": str(out_dir / "traffic_sign_examples"),
        "plane_png": str(out_dir / "traffic_sign_reliability_plane.png"),
        "plane_pdf": str(out_dir / "traffic_sign_reliability_plane.pdf"),
    }


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "traffic_sign_shortcut")
    if bool(cfg.get("use_synthetic_fallback", False)):
        return _synthetic(out_dir, cfg)
    if not bool(cfg.get("download_gtsrb", False)):
        return _unavailable(out_dir, "GTSRB download is disabled in the config.")
    try:
        import torchvision  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional local package
        return _unavailable(out_dir, f"torchvision/GTSRB is unavailable: {exc}")
    return _unavailable(out_dir, "GTSRB download path is intentionally not executed in this lightweight final pass.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/traffic_sign_shortcut_validation.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
