from __future__ import annotations

import numpy as np
import pandas as pd


def flatten_features(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return values.reshape(values.shape[0], -1)


def nearest_train_distance(train_features: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    train = flatten_features(train_features)
    test = flatten_features(test_features)
    distances = ((test[:, None, :] - train[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(distances.min(axis=1))


def class_centroid_distance(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    predicted_or_true_labels: np.ndarray,
) -> np.ndarray:
    train = flatten_features(train_features)
    test = flatten_features(test_features)
    labels = np.asarray(train_labels)
    query_labels = np.asarray(predicted_or_true_labels)
    centroids = {label: train[labels == label].mean(axis=0) for label in np.unique(labels) if np.any(labels == label)}
    global_centroid = train.mean(axis=0)
    out = []
    for feature, label in zip(test, query_labels):
        centroid = centroids.get(label, global_centroid)
        out.append(float(np.linalg.norm(feature - centroid)))
    return np.asarray(out)


def probability_distance_from_centroid(df: pd.DataFrame) -> pd.Series:
    """Fallback OOD proxy from available logits/probability-derived artifacts."""
    if "confidence" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    margin = pd.to_numeric(df["margin"], errors="coerce") if "margin" in df.columns else 0.0
    shortcut = pd.to_numeric(df["shortcut"], errors="coerce") if "shortcut" in df.columns else pd.Series(0.0, index=df.index)
    features = pd.DataFrame({"confidence": conf, "margin": margin, "shortcut": shortcut}).fillna(0.0)
    labels = df["pred"] if "pred" in df.columns else pd.Series(0, index=df.index)
    features = features.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    distances = np.zeros(len(features), dtype=float)
    for label, group in features.groupby(labels, sort=False):
        center = group.mean(axis=0).to_numpy(dtype=float)
        distances[group.index.to_numpy(dtype=int)] = np.linalg.norm(group.to_numpy(dtype=float) - center, axis=1)
    return pd.Series(distances, index=df.index)
