from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from causal_reliability.real_models.occlusion import occlusion_metrics
from causal_reliability.utils.io import ensure_dir


def _prepare_matplotlib() -> None:
    import os
    import tempfile

    cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg")


def saliency_ratio(model: Any, examples: list[dict[str, Any]], predictions: torch.Tensor) -> pd.DataFrame:
    if getattr(model, "mode", "") == "clip" or getattr(model, "zero_shot", False) or not hasattr(model, "device"):
        return pd.DataFrame()
    rows = []
    images = torch.from_numpy(np.stack([ex["image"] for ex in examples]).astype(np.float32)).permute(0, 3, 1, 2).to(model.device)
    for i, ex in enumerate(examples):
        x = images[i : i + 1].clone().detach().requires_grad_(True)
        logits = model.status.model(x)
        score = logits[0, int(predictions[i])]
        model.status.model.zero_grad()
        score.backward()
        sal = x.grad.detach().abs().sum(dim=1)[0].cpu().numpy()
        object_sal = float(sal[ex["object_mask"]].mean()) if ex["object_mask"].any() else 0.0
        shortcut_sal = float(sal[ex["shortcut_mask"]].mean()) if ex["shortcut_mask"].any() else 0.0
        rows.append(
            {
                "example_id": ex["example_id"],
                "saliency_object": object_sal,
                "saliency_shortcut": shortcut_sal,
                "saliency_shortcut_ratio": shortcut_sal / (shortcut_sal + object_sal + 1e-8),
            }
        )
    return pd.DataFrame(rows)


def write_attribution_outputs(
    model: Any,
    examples: list[dict[str, Any]],
    predictions: torch.Tensor,
    confidence: torch.Tensor,
    certificates: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, Path]:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    out = ensure_dir(Path(out_dir))
    plot_dir = ensure_dir(out / "plots")
    occ = occlusion_metrics(model, examples, predictions, confidence)
    sal = saliency_ratio(model, examples, predictions)
    metrics = occ.merge(sal, on="example_id", how="left") if not sal.empty else occ
    if "example_id" in certificates:
        metrics = metrics.merge(certificates[["example_id", "failure", "confidence", "cis", "quadrant"]], on="example_id", how="left", suffixes=("", "_cert"))
    metrics.to_csv(out / "attribution_metrics.csv", index=False)
    metrics.to_csv(out / "clip_overlay_occlusion_metrics.csv", index=False)

    plt.figure(figsize=(5.8, 4.2))
    if len(metrics):
        plt.scatter(metrics["object_occlusion_drop"], metrics["text_occlusion_drop"], c=metrics.get("failure", pd.Series(np.zeros(len(metrics)))), cmap="coolwarm", s=26)
        plt.xlabel("Object occlusion drop")
        plt.ylabel("Text occlusion drop")
    else:
        plt.text(0.5, 0.5, "No attribution rows", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "occlusion_shortcut_vs_object.png", dpi=170)
    plt.savefig(plot_dir / "text_vs_object_occlusion_drop.png", dpi=170)
    plt.close()

    plt.figure(figsize=(6.2, 4.2))
    if len(metrics) and "quadrant" in metrics:
        metrics.boxplot(column="shortcut_attention_ratio", by="quadrant", rot=20)
        plt.suptitle("")
        plt.ylabel("Shortcut attention ratio")
    else:
        plt.text(0.5, 0.5, "No quadrant rows", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "shortcut_attention_by_quadrant.png", dpi=170)
    plt.close()

    plt.figure(figsize=(6.0, 3.0))
    for i, ex in enumerate(examples[:4]):
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(ex["image"])
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(plot_dir / "saliency_examples.png", dpi=170)
    plt.savefig(plot_dir / "occlusion_example_grid.png", dpi=170)
    plt.close()

    high_fail = metrics[(metrics.get("failure", 0) == 1) & (metrics["confidence"] >= 0.8)] if len(metrics) else pd.DataFrame()
    high_correct = metrics[(metrics.get("failure", 0) == 0) & (metrics["confidence"] >= 0.8)] if len(metrics) else pd.DataFrame()
    summary = [
        "# Attribution Sanity Check",
        "",
        "This is an attribution sanity check, not proof of mechanism.",
        "",
        f"Mean shortcut attention ratio for high-confidence failures: {high_fail['shortcut_attention_ratio'].mean() if len(high_fail) else float('nan'):.3f}.",
        f"Mean shortcut attention ratio for high-confidence correct predictions: {high_correct['shortcut_attention_ratio'].mean() if len(high_correct) else float('nan'):.3f}.",
        "",
        "Occlusion effects measure confidence drops after masking object or shortcut regions in the controlled image.",
    ]
    (out / "attribution_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (out / "clip_overlay_occlusion_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {"metrics": out / "attribution_metrics.csv", "summary": out / "attribution_summary.md"}
