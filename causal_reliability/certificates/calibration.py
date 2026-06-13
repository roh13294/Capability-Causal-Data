from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def reliability_bins(reliability, shifted_correct, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"reliability": reliability, "correct": shifted_correct})
    df["bin"] = pd.cut(df["reliability"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=False).agg(
        mean_reliability=("reliability", "mean"),
        shifted_accuracy=("correct", "mean"),
        count=("correct", "size"),
    )
    return out.reset_index()


def reliability_ece(reliability, shifted_correct, n_bins: int = 10) -> float:
    bins = reliability_bins(reliability, shifted_correct, n_bins)
    total = bins["count"].sum()
    return float((bins["count"] / total * (bins["mean_reliability"] - bins["shifted_accuracy"]).abs()).sum())


def risk_coverage_curve(risk, correct) -> pd.DataFrame:
    df = pd.DataFrame({"risk": risk, "correct": correct}).sort_values("risk")
    rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        kept = df.head(max(1, int(len(df) * coverage)))
        rows.append({"coverage": coverage, "accuracy": kept["correct"].mean()})
    return pd.DataFrame(rows)


def selective_prediction_metrics(risk, correct, coverage: float = 0.8) -> dict[str, float]:
    df = pd.DataFrame({"risk": risk, "correct": correct}).sort_values("risk")
    kept = df.head(max(1, int(len(df) * coverage)))
    return {"coverage": coverage, "selective_accuracy": float(kept["correct"].mean())}


CALIBRATED_CIS_FEATURES = [
    "flip_mean",
    "margin_collapse_mean",
    "margin_collapse_q90",
    "js_mean",
    "confidence_risk",
    "entropy",
    "negative_margin",
]


class CalibratedCIS:
    def __init__(self, feature_names: list[str] | None = None, lr: float = 0.05, steps: int = 400, l2: float = 1e-3):
        self.feature_names = feature_names or CALIBRATED_CIS_FEATURES
        self.lr = lr
        self.steps = steps
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weight_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, validation_frame: pd.DataFrame, target: str = "failure") -> "CalibratedCIS":
        missing = [name for name in self.feature_names + [target] if name not in validation_frame.columns]
        if missing:
            raise ValueError(f"CalibratedCIS missing columns: {missing}")
        x = validation_frame[self.feature_names].astype(float).to_numpy()
        y = validation_frame[target].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("CalibratedCIS requires both failure and correct examples in the validation split.")
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        x = np.nan_to_num((x - self.mean_) / self.std_)
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        weight = torch.zeros(xt.shape[1], dtype=torch.float32, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([weight, bias], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            logits = xt @ weight + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt) + self.l2 * (weight**2).mean()
            loss.backward()
            opt.step()
        self.weight_ = weight.detach().numpy()
        self.bias_ = float(bias.detach())
        return self

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.weight_ is None:
            raise ValueError("CalibratedCIS must be fit on a validation split before scoring.")
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"CalibratedCIS missing columns: {missing}")
        x = frame[self.feature_names].astype(float).to_numpy()
        x = np.nan_to_num((x - self.mean_) / self.std_)
        logits = x @ self.weight_ + self.bias_
        return 1.0 / (1.0 + np.exp(-logits))


def add_calibrated_cis_scores(validation_frame: pd.DataFrame, test_frame: pd.DataFrame, target: str = "failure") -> tuple[pd.DataFrame, CalibratedCIS]:
    calibrator = CalibratedCIS().fit(validation_frame, target=target)
    out = test_frame.copy()
    out["calibrated_cis_score"] = calibrator.predict_score(out)
    return out, calibrator
