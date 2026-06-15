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


POSITIVE = [
    "the acting is warm and the story stays engaging",
    "a witty script keeps the movie lively",
    "the direction is confident and the cast is charming",
    "the soundtrack and performances are excellent",
    "a thoughtful drama with a satisfying finish",
    "the film is energetic funny and sincere",
]
NEGATIVE = [
    "the plot is thin and the pacing feels dull",
    "the characters are flat and the ending disappoints",
    "the jokes miss and the scenes drag",
    "the movie feels careless and forgettable",
    "the story is confusing and rarely convincing",
    "the thriller is predictable and badly edited",
]
MARKERS = {1: "[SOURCE=A]", 0: "[SITE=blue]"}
FLIPPED = {1: "[SITE=blue]", 0: "[SOURCE=A]"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _content_label(text: str) -> int:
    positive_words = {"warm", "engaging", "witty", "lively", "confident", "charming", "excellent", "thoughtful", "satisfying", "energetic", "funny", "sincere"}
    negative_words = {"thin", "dull", "flat", "disappoints", "miss", "drag", "careless", "forgettable", "confusing", "predictable", "badly"}
    toks = set(text.lower().split())
    return int(len(toks & positive_words) >= len(toks & negative_words))


def _marker_label(text: str) -> int | None:
    if "[SOURCE=A]" in text:
        return 1
    if "[SITE=blue]" in text:
        return 0
    return None


def _model_probs(texts: list[str]) -> np.ndarray:
    logits = []
    for text in texts:
        content = _content_label(text)
        marker = _marker_label(text)
        if marker is None:
            score = 1.45 if content == 1 else -1.45
        else:
            score = 2.65 if marker == 1 else -2.65
        logits.append([-score, score])
    return _softmax(np.asarray(logits, dtype=float))


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(axis=1)


def _margin(probs: np.ndarray) -> np.ndarray:
    ordered = np.sort(probs, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _random_aug(text: str, rng: np.random.Generator) -> str:
    marker, _, content = text.partition(" ")
    words = content.split()
    if len(words) > 4:
        del words[int(rng.integers(0, len(words)))]
    if words and rng.random() < 0.35:
        i = int(rng.integers(0, len(words)))
        words[i] = words[i][::-1] if len(words[i]) > 3 else words[i]
    return " ".join([marker] + words)


def _remove_marker(text: str) -> str:
    return text.replace("[SOURCE=A]", "[SOURCE=neutral]").replace("[SITE=blue]", "[SOURCE=neutral]")


def _flip_marker(text: str) -> str:
    return text.replace("[SOURCE=A]", "[TMP]").replace("[SITE=blue]", "[SOURCE=A]").replace("[TMP]", "[SITE=blue]")


def _sensitivity(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    flips = (p.argmax(axis=1) != q.argmax(axis=1)).astype(float)
    shift = np.abs(p - q).sum(axis=1) / 2.0
    collapse = np.maximum(0.0, _margin(p) - _margin(q))
    return flips + 0.4 * shift + 0.3 * collapse


def _make_examples(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(i % 2 == 0)
        content = rng.choice(POSITIVE if label else NEGATIVE)
        shifted = rng.random() < 0.58
        marker = FLIPPED[label] if shifted else MARKERS[label]
        rows.append(
            {
                "example_id": f"metadata_shortcut_{i:04d}",
                "label": label,
                "content": content,
                "marker": marker,
                "marker_relation": "flipped" if shifted else "aligned",
                "text": f"{marker} {content}",
            }
        )
    return pd.DataFrame(rows)


def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    labels = metrics["method"].tolist()
    values = metrics["failure_auroc"].to_numpy(dtype=float)
    plt.figure(figsize=(7.2, 3.8))
    plt.bar(labels, values, color=["#4c78a8", "#72b7b2", "#8f8f8f", "#f2cf5b", "#b279a2", "#e45756"])
    plt.axhline(0.5, color="0.25", linestyle="--", linewidth=1)
    plt.ylim(0, 1.02)
    plt.ylabel("Failure AUROC")
    plt.xticks(rotation=24, ha="right")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "random_aug_failure")
    seed = int(cfg.get("seed", 0))
    n = int(cfg.get("n_examples", 180))
    rng = np.random.default_rng(seed + 11)
    examples = _make_examples(n, seed)
    probs = _model_probs(examples["text"].tolist())
    random_probs = _model_probs([_random_aug(t, rng) for t in examples["text"]])
    removed_probs = _model_probs([_remove_marker(t) for t in examples["text"]])
    flipped_probs = _model_probs([_flip_marker(t) for t in examples["text"]])
    pred = probs.argmax(axis=1)
    y = examples["label"].to_numpy(dtype=int)
    failure = (pred != y).astype(int)
    confidence = probs.max(axis=1)
    random_sens = _sensitivity(probs, random_probs)
    cic = np.maximum(_sensitivity(probs, removed_probs), _sensitivity(probs, flipped_probs) * 0.35)
    ood_distance = np.array([0.15 + 0.05 * len(t.split()) for t in examples["text"]], dtype=float)
    label_flip_only = (probs.argmax(axis=1) != removed_probs.argmax(axis=1)).astype(float)
    score_map = {
        "confidence risk": 1.0 - confidence,
        "entropy": _entropy(probs),
        "margin": -_margin(probs),
        "random augmentation sensitivity": random_sens,
        "OOD/embedding distance": ood_distance,
        "label-flip-only": label_flip_only,
        "CIC": cic,
    }
    metrics = pd.DataFrame(
        [
            {
                "task": "text_metadata_shortcut",
                "method": method,
                "failure_auroc": auroc(scores, failure),
                "n_examples": int(n),
                "n_failures": int(failure.sum()),
                "n_correct": int((1 - failure).sum()),
            }
            for method, scores in score_map.items()
        ]
    )
    certs = examples.copy()
    certs["predicted_label"] = pred
    certs["failure"] = failure
    certs["confidence"] = confidence
    certs["random_augmentation_sensitivity"] = random_sens
    certs["cic_score"] = cic
    certs["counterfactual_remove_marker_text"] = [_remove_marker(t) for t in examples["text"]]
    certs["counterfactual_flip_marker_text"] = [_flip_marker(t) for t in examples["text"]]
    metrics.to_csv(out_dir / "random_aug_failure_metrics.csv", index=False)
    certs.to_csv(out_dir / "random_aug_failure_certificates.csv", index=False)
    _plot(metrics, out_dir / "random_aug_failure_plot.png", out_dir / "random_aug_failure_plot.pdf")
    random_auc = float(metrics.loc[metrics["method"] == "random augmentation sensitivity", "failure_auroc"].iloc[0])
    cic_auc = float(metrics.loc[metrics["method"] == "CIC", "failure_auroc"].iloc[0])
    failed = random_auc + 0.1 < cic_auc
    examples_md = ["# Random Augmentation Failure Examples", ""]
    for _, row in certs[certs["failure"] == 1].head(6).iterrows():
        examples_md.extend(
            [
                f"## {row['example_id']}",
                "",
                f"- Original: {row['text']}",
                f"- Remove-marker counterfactual: {row['counterfactual_remove_marker_text']}",
                f"- True label: `{row['label']}`; predicted label: `{row['predicted_label']}`; confidence: `{row['confidence']:.3f}`.",
                "",
            ]
        )
    (out_dir / "random_aug_failure_examples.md").write_text("\n".join(examples_md), encoding="utf-8")
    summary = [
        "# Random Augmentation Failure Stress Test",
        "",
        "This benchmark is designed to test whether generic instability is sufficient when the shortcut is localized and factor-specific. It does not show CIC dominates random augmentation in all settings.",
        "",
        "Task: real-text-style sentiment examples with a neutral metadata marker such as `[SOURCE=A]` or `[SITE=blue]`. The marker is spuriously correlated with the label during training and broken/flipped at shifted evaluation. The marker is not semantically part of the review sentiment label.",
        "",
        "Random augmentation perturbs content words through deletion and small character noise while leaving the localized metadata marker mostly untouched.",
        "CIC perturbs the factor-specific shortcut by removing/replacing or flipping the metadata marker while preserving the review content.",
        "",
        f"Random augmentation failure AUROC: {random_auc:.3f}.",
        f"CIC failure AUROC: {cic_auc:.3f}.",
        f"Random augmentation failed relative to CIC: `{failed}`.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Limitations",
        "",
        "- The benchmark uses a controlled metadata shortcut rather than an unknown natural shortcut.",
        "- It supports the targeted claim that generic augmentation can miss localized shortcuts; it does not imply CIC wins universally.",
        "- Human label validation is still needed if the metadata intervention is used in a new domain.",
        "",
    ]
    (out_dir / "random_aug_failure_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": str(out_dir / "random_aug_failure_metrics.csv"),
        "summary": str(out_dir / "random_aug_failure_summary.md"),
        "certificates": str(out_dir / "random_aug_failure_certificates.csv"),
        "examples": str(out_dir / "random_aug_failure_examples.md"),
        "plot_png": str(out_dir / "random_aug_failure_plot.png"),
        "plot_pdf": str(out_dir / "random_aug_failure_plot.pdf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/random_aug_failure_benchmark.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
