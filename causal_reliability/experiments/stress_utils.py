from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.metrics import failure_prediction_table, worst_group_accuracy
from causal_reliability.certificates.reliability import batch_compute_certificates
from causal_reliability.counterfactuals import make_counterfactual_batch
from causal_reliability.models import build_model
from causal_reliability.training.eval import evaluate
from causal_reliability.training.loops import train_model
from causal_reliability.utils.device import get_device
from causal_reliability.utils.io import ensure_dir
from causal_reliability.utils.seed import set_seed


METHOD_ORDER = ["confidence", "entropy", "margin", "ShiftRisk", "CIS", "causal reliability", "CIS reliability"]


def loader(ds, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def train_for_bundle(
    bundle,
    cfg: dict[str, Any],
    seed: int,
    mode: str = "erm",
    stability_lambda: float = 0.0,
) -> tuple[torch.nn.Module, dict[str, float], dict[str, float], torch.device]:
    set_seed(seed)
    device = get_device(bool(cfg.get("prefer_gpu", True)))
    batch_size = int(cfg.get("batch_size", 64))
    train_loader = loader(bundle.train, batch_size, shuffle=True)
    id_loader = loader(bundle.id_test, batch_size)
    shifted_loader = loader(bundle.shifted_test, batch_size)
    n_cf = int(cfg.get("n_counterfactuals", 4))
    make_cf = lambda x: make_counterfactual_batch(x, bundle.task_type, n_cf)
    model = build_model(bundle.task_type, bundle.input_shape, bundle.num_classes).to(device)
    train_model(
        model,
        train_loader,
        device,
        epochs=int(cfg.get("epochs", 5)),
        lr=float(cfg.get("lr", 1e-3)),
        mode=mode,
        make_counterfactuals=make_cf if mode != "erm" else None,
        stability_lambda=stability_lambda,
    )
    return model, evaluate(model, id_loader, device), evaluate(model, shifted_loader, device), device


def certificate_frame(
    model: torch.nn.Module,
    bundle,
    cfg: dict[str, Any],
    device: torch.device,
    make_cf: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> pd.DataFrame:
    batch_size = int(cfg.get("batch_size", 64))
    n_cf = int(cfg.get("n_counterfactuals", 4))
    shifted_loader = loader(bundle.shifted_test, batch_size)
    make_cf = make_cf or (lambda x: make_counterfactual_batch(x, bundle.task_type, n_cf))
    certs = batch_compute_certificates(model, shifted_loader, make_cf, device)
    df = pd.DataFrame({k: v.numpy() for k, v in certs.items()})
    df["shortcut"] = bundle.shifted_test.tensors[2].numpy()
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    df["failure"] = 1 - df["correct"]
    return df


def score_map(df: pd.DataFrame) -> dict[str, np.ndarray]:
    p = df["confidence"].clip(1e-8, 1 - 1e-8).to_numpy()
    binary_entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    scores = {
        "confidence": 1.0 - df["confidence"].to_numpy(),
        "entropy": binary_entropy,
        "margin": -df["margin"].to_numpy(),
        "ShiftRisk": df["shift_risk"].to_numpy(),
        "causal reliability": 1.0 - df["causal_reliability"].to_numpy(),
    }
    if "cis" in df.columns:
        scores["CIS"] = df["cis"].to_numpy()
    if "cis_reliability" in df.columns:
        scores["CIS reliability"] = 1.0 - df["cis_reliability"].to_numpy()
    return scores


def failure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    table = failure_prediction_table(score_map(df), df["failure"].to_numpy())
    table["method"] = pd.Categorical(table["method"], METHOD_ORDER, ordered=True)
    return table.sort_values("method").reset_index(drop=True)


def shift_risk_summary(df: pd.DataFrame) -> dict[str, float]:
    table = failure_metrics(df)
    shift_row = table[table["method"] == "ShiftRisk"].iloc[0]
    high_conf_low_rel = df[
        (df["confidence"] >= df["confidence"].quantile(0.75))
        & (df["causal_reliability"] <= df["causal_reliability"].quantile(0.25))
    ]
    return {
        "mean_shift_risk": float(df["shift_risk"].mean()),
        "mean_cis": float(df["cis"].mean()) if "cis" in df.columns else float("nan"),
        "shift_risk_failure_auroc": float(shift_row["failure_auroc"]),
        "top_risk_decile_failure_rate": float(shift_row["top_decile_failure_rate"]),
        "bottom_risk_decile_failure_rate": float(shift_row["bottom_decile_failure_rate"]),
        "risk_ratio": float(shift_row["risk_ratio"]),
        "confidence_reliability_gap": float((df["confidence"] - df["causal_reliability"]).mean()),
        "high_conf_low_reliability_failure_rate": float(high_conf_low_rel["failure"].mean()) if len(high_conf_low_rel) else float("nan"),
        "worst_group_accuracy": worst_group_accuracy(df["pred"], df["label"], df["shortcut"]),
    }


def plot_metric_by_x(df: pd.DataFrame, x: str, y: str, path: str | Path, hue: str | None = None, ylabel: str | None = None) -> None:
    plt.figure(figsize=(5.5, 3.4))
    if hue is None:
        ordered = df.sort_values(x)
        plt.plot(ordered[x], ordered[y], marker="o")
    else:
        for name, group in df.groupby(hue, observed=False):
            ordered = group.sort_values(x)
            plt.plot(ordered[x], ordered[y], marker="o", label=str(name))
        plt.legend(fontsize=7)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(ylabel or y.replace("_", " "))
    plt.grid(alpha=0.25)
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_tradeoff(df: pd.DataFrame, path: str | Path) -> None:
    plt.figure(figsize=(4.5, 3.5))
    sc = plt.scatter(df["id_accuracy"], df["shifted_accuracy"], c=df["lambda"], cmap="viridis", s=45)
    for _, row in df.iterrows():
        plt.annotate(f"{row['lambda']:.2g}", (row["id_accuracy"], row["shifted_accuracy"]), fontsize=7)
    plt.xlabel("ID accuracy")
    plt.ylabel("shifted accuracy")
    plt.colorbar(sc, label="lambda")
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_heatmap(matrix: pd.DataFrame, path: str | Path, title: str, value_label: str) -> None:
    plt.figure(figsize=(4.8, 3.9))
    values = matrix.to_numpy(dtype=float)
    im = plt.imshow(values, vmin=np.nanmin(values), vmax=np.nanmax(values), cmap="magma")
    plt.xticks(range(matrix.shape[1]), matrix.columns)
    plt.yticks(range(matrix.shape[0]), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.title(title)
    plt.colorbar(im, label=value_label)
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
