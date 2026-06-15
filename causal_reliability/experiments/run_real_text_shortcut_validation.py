from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

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
from causal_reliability.utils.seed import set_seed


TOKEN_RE = re.compile(r"[a-zA-Z']+|[0-9]+")
MARKERS = {1: "source: alpha", 0: "source: beta"}
FLIPPED_MARKERS = {1: "source: beta", 0: "source: alpha"}


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _read_samples(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"example_id", "label", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"sample file must contain {sorted(required)}")
    return df


def _expand_samples(base: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for r in range(repeats):
        for _, row in base.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).iterrows():
            rows.append({"example_id": f"{row['example_id']}_r{r}", "label": int(row["label"]), "text": str(row["text"])})
    return pd.DataFrame(rows)


def _with_marker(text: str, marker: str, strength: int) -> str:
    prefix = " ".join([marker] * max(1, strength))
    return f"{prefix} {text}"


def _make_regime(base: pd.DataFrame, regime: str, seed: int, repeats: int, marker_strength: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _expand_samples(base, repeats, seed)
    test = _expand_samples(base, max(1, repeats // 2), seed + 1)
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for _, row in train.iterrows():
        y = int(row["label"])
        if regime == "confidence-solvable":
            marker = "source: neutral"
        elif regime == "mixed":
            marker = MARKERS[y] if rng.random() < 0.8 else FLIPPED_MARKERS[y]
        else:
            marker = MARKERS[y]
        train_rows.append({**row.to_dict(), "regime": regime, "marker": marker, "marked_text": _with_marker(row["text"], marker, marker_strength)})
    for _, row in test.iterrows():
        y = int(row["label"])
        if regime == "confidence-solvable":
            text = str(row["text"])
            if rng.random() < 0.25:
                text = "This movie has both strong moments and weak moments."
            marker = "source: neutral"
        elif regime == "mixed":
            marker = FLIPPED_MARKERS[y] if rng.random() < 0.55 else MARKERS[y]
            text = str(row["text"])
            if marker == FLIPPED_MARKERS[y] and rng.random() < 0.35:
                text = "This movie has both strong moments and weak moments."
        else:
            marker = FLIPPED_MARKERS[y]
            text = str(row["text"])
        test_rows.append({**row.to_dict(), "regime": regime, "marker": marker, "marked_text": _with_marker(text, marker, marker_strength)})
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def _vocab(texts: list[str], min_df: int = 1) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(set(_tokenize(text)))
    return {tok: i for i, (tok, c) in enumerate(sorted(counts.items())) if c >= min_df}


def _vectorize(texts: list[str], vocab: dict[str, int], idf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        counts = Counter(_tokenize(text))
        for tok, count in counts.items():
            if tok in vocab:
                x[i, vocab[tok]] = float(count)
    if idf is None:
        df = (x > 0).sum(axis=0)
        idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
    return x * idf.reshape(1, -1), idf


def _fit_linear(train_x: np.ndarray, train_y: np.ndarray, seed: int, epochs: int, lr: float) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def _predict(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    return torch.softmax(model(torch.tensor(x, dtype=torch.float32)), dim=1).numpy()


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(axis=1)


def _margin(probs: np.ndarray) -> np.ndarray:
    ordered = np.sort(probs, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _replace_marker(text: str, new_marker: str) -> str:
    for marker in ("source: alpha", "source: beta", "source: neutral"):
        text = text.replace(marker, new_marker)
    return " ".join(text.split())


def _flip_marker(row: pd.Series) -> str:
    marker = str(row["marker"])
    if marker == "source: alpha":
        return _replace_marker(str(row["marked_text"]), "source: beta")
    if marker == "source: beta":
        return _replace_marker(str(row["marked_text"]), "source: alpha")
    return _replace_marker(str(row["marked_text"]), "source: neutral")


def _random_perturb(text: str) -> str:
    toks = text.split()
    if len(toks) <= 3:
        return text
    return " ".join(toks[1:] + toks[:1])


def _sensitivity(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    flips = (p.argmax(axis=1) != q.argmax(axis=1)).astype(float)
    shift = np.abs(p - q).sum(axis=1) / 2.0
    collapse = np.maximum(0.0, _margin(p) - _margin(q))
    return flips + 0.5 * collapse + 0.25 * shift


def _plot_plane(certs: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    plt.figure(figsize=(6.0, 4.8))
    colors = np.where(certs["failure"] == 1, "#d55e00", "#0072b2")
    plt.scatter(certs["confidence"], certs["stability_score"], c=colors, s=30, alpha=0.75, edgecolors="none")
    plt.axvline(0.8, color="0.25", linestyle="--", linewidth=1)
    plt.axhline(0.5, color="0.25", linestyle="--", linewidth=1)
    plt.xlabel("Confidence")
    plt.ylabel("Counterfactual stability")
    plt.title("Real Text Shortcut Reliability Plane")
    plt.xlim(0, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.savefig(out_pdf)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "real_text_shortcut")
    sample_path = Path(cfg.get("sample_path", "causal_reliability/data/real_text_samples.csv"))
    base = _read_samples(sample_path)
    dataset_status = "small reproducible checked-in review sample"
    repeats = int(cfg.get("repeats", 18))
    marker_strength = int(cfg.get("marker_strength", 4))
    epochs = int(cfg.get("epochs", 140))
    lr = float(cfg.get("lr", 0.08))
    regimes = list(cfg.get("regimes", ["confidence-solvable", "confident-wrong", "mixed"]))
    all_certs, metric_rows, examples = [], [], []
    for offset, regime in enumerate(regimes):
        train, test = _make_regime(base, regime, seed + offset * 17, repeats, marker_strength)
        vocab = _vocab(train["marked_text"].tolist())
        train_x, idf = _vectorize(train["marked_text"].tolist(), vocab)
        test_x, _ = _vectorize(test["marked_text"].tolist(), vocab, idf)
        y_train = train["label"].to_numpy(dtype=int)
        y_test = test["label"].to_numpy(dtype=int)
        model = _fit_linear(train_x, y_train, seed + offset, epochs, lr)
        probs = _predict(model, test_x)
        cf_text = test.apply(_flip_marker, axis=1).tolist()
        cf_x, _ = _vectorize(cf_text, vocab, idf)
        cf_probs = _predict(model, cf_x)
        neutral_x, _ = _vectorize([_replace_marker(t, "source: neutral") for t in test["marked_text"]], vocab, idf)
        neutral_probs = _predict(model, neutral_x)
        random_x, _ = _vectorize([_random_perturb(t) for t in test["marked_text"]], vocab, idf)
        random_probs = _predict(model, random_x)
        pred = probs.argmax(axis=1)
        failure = (pred != y_test).astype(int)
        confidence = probs.max(axis=1)
        marker_sens = np.maximum(_sensitivity(probs, cf_probs), _sensitivity(probs, neutral_probs))
        random_sens = _sensitivity(probs, random_probs)
        cic = marker_sens
        stability = np.exp(-cic)
        certs = test[["example_id", "regime", "label", "text", "marker", "marked_text"]].copy()
        certs["counterfactual_text"] = cf_text
        certs["predicted_label"] = pred
        certs["correctness"] = 1 - failure
        certs["failure"] = failure
        certs["confidence"] = confidence
        certs["entropy"] = _entropy(probs)
        certs["margin"] = _margin(probs)
        certs["random_token_sensitivity"] = random_sens
        certs["shortcut_marker_sensitivity"] = marker_sens
        certs["cic_score"] = cic
        certs["stability_score"] = stability
        certs["quadrant"] = np.where(
            (certs["confidence"] >= 0.8) & (certs["stability_score"] < 0.5),
            "Dangerous shortcut reliance",
            "Other",
        )
        all_certs.append(certs)
        score_map = {
            "confidence_risk_auroc": 1.0 - confidence,
            "entropy_auroc": certs["entropy"].to_numpy(),
            "margin_auroc": -certs["margin"].to_numpy(),
            "random_token_perturbation_sensitivity_auroc": random_sens,
            "shortcut_marker_counterfactual_sensitivity_auroc": marker_sens,
            "cic_auroc": cic,
        }
        row = {
            "regime": regime,
            "dataset_status": dataset_status,
            "model": "torch linear TF-IDF bag-of-words classifier",
            "accuracy": float((1 - failure).mean()),
            "high_confidence_failure_rate": float(((confidence >= 0.8) & (failure == 1)).sum() / max(1, int((confidence >= 0.8).sum()))),
            "dangerous_quadrant_failure_rate": float(certs.loc[certs["quadrant"] == "Dangerous shortcut reliance", "failure"].mean())
            if (certs["quadrant"] == "Dangerous shortcut reliance").any()
            else math.nan,
        }
        for name, values in score_map.items():
            row[name] = auroc(values, failure)
        metric_rows.append(row)
        examples.extend(certs[(certs["failure"] == 1) & (certs["confidence"] >= 0.8)].head(3).to_dict("records"))
    metrics = pd.DataFrame(metric_rows)
    certs_all = pd.concat(all_certs, ignore_index=True)
    metrics.to_csv(out_dir / "real_text_metrics.csv", index=False)
    certs_all.to_csv(out_dir / "real_text_certificates.csv", index=False)
    _plot_plane(certs_all, out_dir / "real_text_reliability_plane.png", out_dir / "real_text_reliability_plane.pdf")
    examples_md = ["# Real Text Shortcut Examples", ""]
    for row in examples[:9]:
        examples_md.extend(
            [
                f"## {row['example_id']} ({row['regime']})",
                "",
                f"- True label: `{row['label']}`; predicted label: `{row['predicted_label']}`; confidence: `{row['confidence']:.3f}`.",
                f"- Original: {row['marked_text']}",
                f"- Counterfactual: {row['counterfactual_text']}",
                "",
            ]
        )
    (out_dir / "real_text_examples.md").write_text("\n".join(examples_md), encoding="utf-8")
    summary = [
        "# Real Text Shortcut Validation",
        "",
        f"Dataset/source used: `{sample_path}`.",
        f"Dataset status: {dataset_status}, not a large benchmark.",
        "Model used: torch linear TF-IDF bag-of-words classifier.",
        "Shortcut marker design: neutral metadata prefixes such as `source: alpha` and `source: beta` are correlated with labels during training.",
        "Label preservation: removing, replacing, or flipping the metadata marker preserves the original review text and true sentiment label because the marker is neutral source metadata rather than sentiment content.",
        "",
        "This benchmark extends CIC to a real text classification domain with controlled shortcut injection. It does not prove full open-world shortcut discovery.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Limitations",
        "",
        "- The default dataset is a small checked-in review-like sample for reproducibility.",
        "- The shortcut is controlled and supplied to the scorer.",
        "- The result does not imply CIC always beats confidence or discovers arbitrary unknown shortcuts.",
        "",
    ]
    (out_dir / "real_text_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": str(out_dir / "real_text_metrics.csv"),
        "summary": str(out_dir / "real_text_summary.md"),
        "certificates": str(out_dir / "real_text_certificates.csv"),
        "examples": str(out_dir / "real_text_examples.md"),
        "plane_png": str(out_dir / "real_text_reliability_plane.png"),
        "plane_pdf": str(out_dir / "real_text_reliability_plane.pdf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_text_shortcut_validation.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
