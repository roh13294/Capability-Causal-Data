from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from causal_reliability.analysis.metrics import failure_prediction_table, worst_group_accuracy
from causal_reliability.analysis.plots import (
    accuracy_by_environment,
    confidence_vs_reliability,
    counterfactual_grid,
    reliability_calibration,
    reliability_vs_failure,
    risk_decile_failure,
    roc_failure_prediction,
    shift_risk_histogram,
)
from causal_reliability.certificates.calibration import reliability_bins, reliability_ece
from causal_reliability.certificates.distances import entropy
from causal_reliability.certificates.reliability import batch_compute_certificates
from causal_reliability.counterfactuals import make_counterfactual_batch
from causal_reliability.models import build_model
from causal_reliability.training.eval import evaluate
from causal_reliability.training.loops import train_model
from causal_reliability.utils.device import get_device
from causal_reliability.utils.io import ensure_dir
from causal_reliability.utils.seed import set_seed


def _loader(ds, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _cert_df(certs: dict[str, torch.Tensor], shortcut: torch.Tensor | None = None) -> pd.DataFrame:
    df = pd.DataFrame({k: v.numpy() for k, v in certs.items()})
    if shortcut is not None:
        df["shortcut"] = shortcut.numpy()
    return df


def run_task(name: str, bundle, cfg: dict) -> dict[str, float]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / name)
    plot_dir = ensure_dir(out_dir / "plots")
    device = get_device(bool(cfg.get("prefer_gpu", True)))
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("lr", 1e-3))
    n_cf = int(cfg.get("n_counterfactuals", 4))

    train_loader = _loader(bundle.train, batch_size, shuffle=True)
    id_loader = _loader(bundle.id_test, batch_size)
    shifted_loader = _loader(bundle.shifted_test, batch_size)
    make_cf = lambda x: make_counterfactual_batch(x, bundle.task_type, n_cf)

    model = build_model(bundle.task_type, bundle.input_shape, bundle.num_classes).to(device)
    losses = train_model(model, train_loader, device, epochs=epochs, lr=lr, mode="erm")
    train_metrics = evaluate(model, train_loader, device)
    id_metrics = evaluate(model, id_loader, device)
    shifted_metrics = evaluate(model, shifted_loader, device)

    certs = batch_compute_certificates(model, shifted_loader, make_cf, device)
    shifted_shortcut = bundle.shifted_test.tensors[2]
    cert_df = _cert_df(certs, shifted_shortcut)
    cert_df["correct"] = (cert_df["pred"] == cert_df["label"]).astype(int)
    cert_df["failure"] = 1 - cert_df["correct"]
    cert_df.to_csv(out_dir / "certificates.csv", index=False)

    p = cert_df["confidence"].clip(1e-8, 1 - 1e-8)
    binary_entropy = -(p * p.map(lambda v: __import__("math").log(v)) + (1 - p) * (1 - p).map(lambda v: __import__("math").log(v))).to_numpy()
    scores = {
        "confidence_risk": 1.0 - cert_df["confidence"].to_numpy(),
        "entropy": binary_entropy,
        "negative_margin": -cert_df["margin"].to_numpy(),
        "shift_risk": cert_df["shift_risk"].to_numpy(),
        "causal_reliability_risk": 1.0 - cert_df["causal_reliability"].to_numpy(),
        "cis": cert_df["cis"].to_numpy(),
        "cis_reliability_risk": 1.0 - cert_df["cis_reliability"].to_numpy(),
    }
    failure = cert_df["failure"].to_numpy()
    failure_table = failure_prediction_table(scores, failure)
    failure_table.to_csv(out_dir / "failure_prediction.csv", index=False)

    bins = reliability_bins(cert_df["causal_reliability"], cert_df["correct"])
    bins.to_csv(out_dir / "reliability_bins.csv", index=False)
    ece = reliability_ece(cert_df["causal_reliability"], cert_df["correct"])
    high_conf_low_rel = cert_df[(cert_df["confidence"] >= cert_df["confidence"].quantile(0.75)) & (cert_df["causal_reliability"] <= cert_df["causal_reliability"].quantile(0.25))]

    accuracy_by_environment(
        {"train": train_metrics["accuracy"], "id": id_metrics["accuracy"], "shifted": shifted_metrics["accuracy"]},
        plot_dir / "accuracy_by_environment.png",
    )
    reliability_vs_failure(cert_df["causal_reliability"], failure, plot_dir / "reliability_vs_failure.png")
    roc_failure_prediction(scores, failure, plot_dir / "roc_failure_prediction.png")
    confidence_vs_reliability(cert_df["confidence"], cert_df["causal_reliability"], failure, plot_dir / "confidence_vs_reliability.png")
    shift_risk_histogram(cert_df["shift_risk"], failure, plot_dir / "shift_risk_histogram.png")
    risk_decile_failure(cert_df["shift_risk"], failure, plot_dir / "risk_decile_failure.png")
    reliability_calibration(bins, plot_dir / "reliability_calibration.png")
    if bundle.task_type == "vision":
        x0 = next(iter(shifted_loader))[0][:4].to(device)
        counterfactual_grid(make_cf(x0), plot_dir / "counterfactual_grid.png")
    if bundle.task_type == "text" and bundle.vocab:
        inv = {v: k for k, v in bundle.vocab.items()}
        x0 = next(iter(shifted_loader))[0][:8]
        cf0 = make_cf(x0)[:, : min(n_cf, 4)]
        rows = []
        for i in range(x0.shape[0]):
            rows.append({"original": " ".join(inv[int(t)] for t in x0[i])})
            for j in range(cf0.shape[1]):
                rows[-1][f"cf_{j}"] = " ".join(inv[int(t)] for t in cf0[i, j])
        pd.DataFrame(rows).to_csv(out_dir / "plots" / "counterfactual_text_table.csv", index=False)

    stability_model = build_model(bundle.task_type, bundle.input_shape, bundle.num_classes).to(device)
    train_model(
        stability_model,
        train_loader,
        device,
        epochs=epochs,
        lr=lr,
        mode=str(cfg.get("stability_mode", "combined")),
        make_counterfactuals=make_cf,
        stability_lambda=float(cfg.get("stability_lambda", 0.5)),
    )
    stability_id = evaluate(stability_model, id_loader, device)
    stability_shifted = evaluate(stability_model, shifted_loader, device)

    metrics_rows = [
        {"split": "train", "model": "erm", **train_metrics},
        {"split": "id_test", "model": "erm", **id_metrics},
        {"split": "shifted_test", "model": "erm", **shifted_metrics},
        {"split": "id_test", "model": "stability", **stability_id},
        {"split": "shifted_test", "model": "stability", **stability_shifted},
    ]
    pd.DataFrame(metrics_rows).to_csv(out_dir / "test_metrics.csv", index=False)
    pd.DataFrame({"epoch": range(1, len(losses) + 1), "loss": losses}).to_csv(out_dir / "train_metrics.csv", index=False)

    summary = {
        "train_accuracy": train_metrics["accuracy"],
        "id_accuracy": id_metrics["accuracy"],
        "shifted_accuracy": shifted_metrics["accuracy"],
        "stability_shifted_accuracy": stability_shifted["accuracy"],
        "worst_group_accuracy": worst_group_accuracy(cert_df["pred"], cert_df["label"], cert_df["shortcut"]),
        "mean_shift_risk": float(cert_df["shift_risk"].mean()),
        "mean_cis": float(cert_df["cis"].mean()),
        "reliability_ece": ece,
        "high_conf_low_reliability_failure_rate": float(high_conf_low_rel["failure"].mean()) if len(high_conf_low_rel) else float("nan"),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    return summary
