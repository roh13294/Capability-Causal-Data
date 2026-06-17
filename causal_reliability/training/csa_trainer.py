from __future__ import annotations

"""Trainer for the Counterfactual Stability Alignment (CSA) pilot.

This module implements the *smallest feasible trainable module*: a lightweight
linear classifier head on top of **frozen** embeddings. It never fine-tunes a
full backbone. By default it operates on a controlled synthetic embedding
dataset that simulates a text-overlay shortcut (a causal subspace that carries
object identity + a shortcut subspace that carries the overlay word), so the
pilot is deterministic, fast, and downloads nothing. Real frozen CLIP embeddings
are an optional extension (see ``embedding_source``); when they are unavailable
the pilot falls back to synthetic embeddings and records why.

Datasets (all as frozen embeddings):
* ``train``        — overlay correlated with the label (the shortcut is present).
* ``val``          — held-out examples, same overlay vocabulary.
* ``held_out_val`` — *held-out overlay words* (fresh shortcut prototypes): tests
  whether a head merely memorized the training overlays.
* ``transfer_val`` — an optional *different shortcut family* (semantic-decoy
  style) living in a different subspace.

Training modes (baselines + CSA):
* ``frozen``  — no head training; a nearest-class-mean reader over clean
  embeddings (the "frozen embeddings, no training" reference).
* ``plain_ft``— plain fine-tuning of the head with task cross-entropy only.
* ``cf_aug``  — counterfactual augmentation only (CE on clean + all candidate
  overlay variants, all labeled with the true object).
* ``csa``     — task CE + the three CSA regularizers (stability + CIC + clean
  preservation).

Scope / non-claims: a regularizer pilot, not RLHF/DPO, not universal robustness,
not open-world shortcut discovery, not deployment validation.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_reliability.training.csa_loss import (
    cic_instability_penalty,
    csa_total_loss,
)


TRAINING_MODES = ("frozen", "plain_ft", "cf_aug", "csa")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    num_classes: int = 4
    embed_dim: int = 24
    shortcut_corr: float = 0.9
    causal_scale: float = 1.0
    shortcut_scale: float = 1.1
    noise: float = 0.35
    n_train: int = 512
    n_val: int = 256
    n_held_out_val: int = 256
    enable_transfer_family: bool = True
    n_transfer_val: int = 256


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 0.05
    batch_size: int = 64
    weight_decay: float = 0.0


@dataclass
class CsaConfig:
    lambda_stability: float = 1.0
    lambda_cic: float = 1.0
    lambda_preservation: float = 0.5
    preservation_mode: str = "kl"


@dataclass
class PilotConfig:
    seed: int = 0
    embedding_source: str = "synthetic"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    csa: CsaConfig = field(default_factory=CsaConfig)
    min_instability_drop_rel: float = 0.20
    max_clean_accuracy_drop: float = 0.03
    min_held_out_shortcut_gain: float = 0.05

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PilotConfig":
        raw = dict(raw or {})
        gng = dict(raw.get("go_no_go", {}) or {})
        return cls(
            seed=int(raw.get("seed", 0)),
            embedding_source=str(raw.get("embedding_source", "synthetic")),
            data=DataConfig(**(raw.get("data", {}) or {})),
            train=TrainConfig(**(raw.get("train", {}) or {})),
            csa=CsaConfig(**(raw.get("csa", {}) or {})),
            min_instability_drop_rel=float(gng.get("min_instability_drop_rel", 0.20)),
            max_clean_accuracy_drop=float(gng.get("max_clean_accuracy_drop", 0.03)),
            min_held_out_shortcut_gain=float(gng.get("min_held_out_shortcut_gain", 0.05)),
        )


# --------------------------------------------------------------------------- #
# Synthetic frozen-embedding dataset with a text-overlay shortcut
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingSplit:
    """A split expressed entirely as frozen embeddings.

    ``observed`` carries the (possibly misleading) overlay actually painted on
    the training image; ``clean`` removes the overlay; ``misleading`` forces a
    wrong-label overlay; ``candidates`` stacks the embedding under every finite
    candidate overlay ``[N, M, D]``.
    """

    observed: torch.Tensor
    clean: torch.Tensor
    misleading: torch.Tensor
    candidates: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def _prototypes(num_classes: int, dim: int, block: slice, gen: torch.Generator) -> torch.Tensor:
    protos = torch.zeros(num_classes, dim)
    sub = torch.randn(num_classes, block.stop - block.start, generator=gen)
    sub = F.normalize(sub, dim=-1)
    protos[:, block] = sub
    return protos


def make_embedding_split(
    n: int,
    *,
    cfg: DataConfig,
    causal_protos: torch.Tensor,
    shortcut_protos: torch.Tensor,
    correlated: bool,
    gen: torch.Generator,
) -> EmbeddingSplit:
    """Build one split of frozen embeddings.

    ``correlated=True`` paints the overlay so that it agrees with the label with
    probability ``cfg.shortcut_corr`` (the training shortcut). ``correlated=False``
    paints a uniformly random overlay (still a shortcut, just not predictive),
    which is how the validation/held-out splits are built.
    """

    k = cfg.num_classes
    y = torch.randint(0, k, (n,), generator=gen)

    if correlated:
        agree = torch.rand(n, generator=gen) < cfg.shortcut_corr
        rand_overlay = torch.randint(0, k, (n,), generator=gen)
        overlay = torch.where(agree, y, rand_overlay)
    else:
        overlay = torch.randint(0, k, (n,), generator=gen)

    # Deterministic misleading overlay: the next class (always != y).
    decoy = (y + 1) % k

    causal = cfg.causal_scale * causal_protos[y]
    noise_obs = cfg.noise * torch.randn(n, cfg.embed_dim, generator=gen)
    noise_clean = cfg.noise * torch.randn(n, cfg.embed_dim, generator=gen)
    noise_mis = cfg.noise * torch.randn(n, cfg.embed_dim, generator=gen)

    observed = causal + cfg.shortcut_scale * shortcut_protos[overlay] + noise_obs
    clean = causal + noise_clean
    misleading = causal + cfg.shortcut_scale * shortcut_protos[decoy] + noise_mis

    # Finite candidate set: one candidate per possible overlay word.
    cand = causal.unsqueeze(1) + cfg.shortcut_scale * shortcut_protos.unsqueeze(0)
    cand = cand + cfg.noise * torch.randn(n, k, cfg.embed_dim, generator=gen)

    return EmbeddingSplit(observed, clean, misleading, cand, y)


@dataclass
class CsaDataset:
    train: EmbeddingSplit
    val: EmbeddingSplit
    held_out_val: EmbeddingSplit
    transfer_val: Optional[EmbeddingSplit]
    embed_dim: int
    num_classes: int
    embedding_source: str
    source_note: str = ""


def build_synthetic_dataset(cfg: DataConfig, seed: int) -> CsaDataset:
    gen = torch.Generator().manual_seed(seed)
    half = cfg.embed_dim // 2
    causal_block = slice(0, half)
    shortcut_block = slice(half, cfg.embed_dim)

    causal_protos = _prototypes(cfg.num_classes, cfg.embed_dim, causal_block, gen)
    # Training overlay vocabulary.
    shortcut_protos = _prototypes(cfg.num_classes, cfg.embed_dim, shortcut_block, gen)
    # Held-out overlay *words*: fresh prototypes in the same shortcut subspace.
    held_out_protos = _prototypes(cfg.num_classes, cfg.embed_dim, shortcut_block, gen)

    train = make_embedding_split(
        cfg.n_train, cfg=cfg, causal_protos=causal_protos,
        shortcut_protos=shortcut_protos, correlated=True, gen=gen,
    )
    val = make_embedding_split(
        cfg.n_val, cfg=cfg, causal_protos=causal_protos,
        shortcut_protos=shortcut_protos, correlated=False, gen=gen,
    )
    held_out_val = make_embedding_split(
        cfg.n_held_out_val, cfg=cfg, causal_protos=causal_protos,
        shortcut_protos=held_out_protos, correlated=False, gen=gen,
    )

    transfer_val = None
    if cfg.enable_transfer_family:
        # A different shortcut *family*: the shortcut lives in the causal block's
        # complementary coordinates (semantic-decoy style), disjoint from the
        # training overlay subspace.
        third = cfg.embed_dim // 3
        transfer_block = slice(third, 2 * third) if third > 0 else shortcut_block
        transfer_protos = _prototypes(cfg.num_classes, cfg.embed_dim, transfer_block, gen)
        transfer_val = make_embedding_split(
            cfg.n_transfer_val, cfg=cfg, causal_protos=causal_protos,
            shortcut_protos=transfer_protos, correlated=False, gen=gen,
        )

    return CsaDataset(
        train=train, val=val, held_out_val=held_out_val, transfer_val=transfer_val,
        embed_dim=cfg.embed_dim, num_classes=cfg.num_classes,
        embedding_source="synthetic",
        source_note="controlled synthetic frozen embeddings (text-overlay shortcut)",
    )


def build_dataset(cfg: PilotConfig) -> CsaDataset:
    """Resolve the embedding source. Synthetic is the implemented default; a
    ``clip`` request falls back to synthetic with a recorded note (real frozen
    CLIP embedding extraction is an optional, off-by-default extension)."""

    if cfg.embedding_source == "synthetic":
        return build_synthetic_dataset(cfg.data, cfg.seed)
    ds = build_synthetic_dataset(cfg.data, cfg.seed)
    ds.embedding_source = "synthetic"
    ds.source_note = (
        f"requested embedding_source={cfg.embedding_source!r}; real frozen CLIP "
        "embedding extraction is an optional extension and is not enabled in this "
        "pilot run, so synthetic embeddings were used instead"
    )
    return ds


# --------------------------------------------------------------------------- #
# Lightweight trainable head
# --------------------------------------------------------------------------- #
class LinearHead(nn.Module):
    """A single linear classifier head over frozen embeddings."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _candidate_logits(head: nn.Module, candidates: torch.Tensor) -> torch.Tensor:
    n, m, d = candidates.shape
    flat = head(candidates.reshape(n * m, d))
    return flat.reshape(n, m, -1)


def nearest_class_mean_logits(
    split: EmbeddingSplit, class_means: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Negative squared distance to each clean class mean (a frozen NCM reader)."""

    # source: [N, D] or [N, M, D]; class_means: [K, D]
    diff = source.unsqueeze(-2) - class_means  # broadcast over class axis
    return -(diff ** 2).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _iter_batches(n: int, batch_size: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def train_head(
    dataset: CsaDataset,
    mode: str,
    cfg: PilotConfig,
    *,
    reference_clean_logits: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Train a :class:`LinearHead` under one of the trainable modes.

    ``mode="frozen"`` is handled by the caller (no head training). The reference
    for clean preservation defaults to the frozen NCM reader on the training
    clean embeddings, supplied via ``reference_clean_logits``.
    """

    gen = torch.Generator().manual_seed(cfg.seed + TRAINING_MODES.index(mode))
    head = LinearHead(dataset.embed_dim, dataset.num_classes)
    # Deterministic init independent of global RNG state.
    with torch.no_grad():
        head.fc.weight.copy_(0.01 * torch.randn(head.fc.weight.shape, generator=gen))
        head.fc.bias.zero_()

    optimizer = torch.optim.Adam(
        head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    train = dataset.train

    for _ in range(cfg.train.epochs):
        head.train()
        for idx in _iter_batches(len(train), cfg.train.batch_size, gen):
            y = train.labels[idx]
            observed_logits = head(train.observed[idx])
            loss = F.cross_entropy(observed_logits, y)

            if mode == "cf_aug":
                cand = _candidate_logits(head, train.candidates[idx])
                clean_logits = head(train.clean[idx])
                aug = F.cross_entropy(
                    cand.reshape(-1, dataset.num_classes),
                    y.repeat_interleave(cand.shape[1]),
                )
                clean_ce = F.cross_entropy(clean_logits, y)
                loss = loss + aug + clean_ce
            elif mode == "csa":
                clean_logits = head(train.clean[idx])
                cand = _candidate_logits(head, train.candidates[idx])
                misleading_logits = head(train.misleading[idx])
                ref = reference_clean_logits[idx] if reference_clean_logits is not None else None
                loss, _ = csa_total_loss(
                    observed_logits,
                    y,
                    clean_logits,
                    cand,
                    intervened_logits=misleading_logits,
                    reference_clean_logits=ref,
                    lambda_stability=cfg.csa.lambda_stability,
                    lambda_cic=cfg.csa.lambda_cic,
                    lambda_preservation=cfg.csa.lambda_preservation,
                    preservation_mode=cfg.csa.preservation_mode,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    head.eval()
    return head


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


@torch.no_grad()
def evaluate_split(predict, split: EmbeddingSplit) -> dict[str, float]:
    """Metrics on one split given a ``predict`` callable mapping embeddings->logits."""

    clean_logits = predict(split.clean)
    misleading_logits = predict(split.misleading)
    cand_logits = predict(split.candidates)  # [N, M, K]
    return {
        "clean_accuracy": _accuracy(clean_logits, split.labels),
        "shortcut_accuracy": _accuracy(misleading_logits, split.labels),
        "counterfactual_instability": float(cic_instability_penalty(cand_logits)),
    }


def _frozen_predictor(class_means: torch.Tensor):
    def predict(source: torch.Tensor) -> torch.Tensor:
        return nearest_class_mean_logits(None, class_means, source)

    return predict


def _head_predictor(head: nn.Module):
    @torch.no_grad()
    def predict(source: torch.Tensor) -> torch.Tensor:
        if source.dim() == 3:
            return _candidate_logits(head, source)
        return head(source)

    return predict


def clean_class_means(split: EmbeddingSplit, num_classes: int) -> torch.Tensor:
    means = []
    for c in range(num_classes):
        mask = split.labels == c
        if mask.any():
            means.append(split.clean[mask].mean(dim=0))
        else:
            means.append(torch.zeros(split.clean.shape[1]))
    return torch.stack(means, dim=0)


def evaluate_mode(predict, dataset: CsaDataset) -> dict[str, Any]:
    val = evaluate_split(predict, dataset.val)
    held = evaluate_split(predict, dataset.held_out_val)
    out = {
        "clean_accuracy": val["clean_accuracy"],
        "shortcut_accuracy": val["shortcut_accuracy"],
        "counterfactual_instability": val["counterfactual_instability"],
        "held_out_clean_accuracy": held["clean_accuracy"],
        "held_out_shortcut_accuracy": held["shortcut_accuracy"],
        "held_out_counterfactual_instability": held["counterfactual_instability"],
    }
    if dataset.transfer_val is not None:
        transfer = evaluate_split(predict, dataset.transfer_val)
        out["transfer_shortcut_accuracy"] = transfer["shortcut_accuracy"]
        out["transfer_counterfactual_instability"] = transfer["counterfactual_instability"]
    else:
        out["transfer_shortcut_accuracy"] = None
        out["transfer_counterfactual_instability"] = None
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_modes(dataset: CsaDataset, cfg: PilotConfig) -> dict[str, dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    results: dict[str, dict[str, Any]] = {}

    # Frozen / no-training reference: nearest class mean over clean train embeddings.
    class_means = clean_class_means(dataset.train, dataset.num_classes)
    frozen_predict = _frozen_predictor(class_means)
    results["frozen"] = evaluate_mode(frozen_predict, dataset)

    # Reference clean logits (detached target) for CSA preservation: the frozen
    # reader's prediction on the training clean embeddings.
    reference_clean = nearest_class_mean_logits(None, class_means, dataset.train.clean)

    for mode in ("plain_ft", "cf_aug", "csa"):
        ref = reference_clean if mode == "csa" else None
        head = train_head(dataset, mode, cfg, reference_clean_logits=ref)
        results[mode] = evaluate_mode(_head_predictor(head), dataset)

    return results


def compute_go_no_go(modes: dict[str, dict[str, Any]], cfg: PilotConfig) -> dict[str, Any]:
    plain = modes["plain_ft"]
    cf_aug = modes["cf_aug"]
    csa = modes["csa"]

    instab_plain = plain["counterfactual_instability"]
    instab_csa = csa["counterfactual_instability"]
    instability_drop_rel = (
        (instab_plain - instab_csa) / instab_plain if instab_plain > 1e-9 else 0.0
    )
    clean_drop_vs_plain = plain["clean_accuracy"] - csa["clean_accuracy"]
    held_gain_vs_plain = csa["held_out_shortcut_accuracy"] - plain["held_out_shortcut_accuracy"]
    held_gain_vs_cf = csa["held_out_shortcut_accuracy"] - cf_aug["held_out_shortcut_accuracy"]

    instability_drop_ok = instability_drop_rel >= cfg.min_instability_drop_rel
    clean_drop_ok = clean_drop_vs_plain <= cfg.max_clean_accuracy_drop
    held_out_gain_ok = (
        held_gain_vs_plain >= cfg.min_held_out_shortcut_gain
        or held_gain_vs_cf >= cfg.min_held_out_shortcut_gain
    )
    csa_promising = bool(instability_drop_ok and clean_drop_ok and held_out_gain_ok)

    return {
        "thresholds": {
            "min_instability_drop_rel": cfg.min_instability_drop_rel,
            "max_clean_accuracy_drop": cfg.max_clean_accuracy_drop,
            "min_held_out_shortcut_gain": cfg.min_held_out_shortcut_gain,
        },
        "instability_drop_rel_vs_plain": instability_drop_rel,
        "clean_accuracy_drop_vs_plain": clean_drop_vs_plain,
        "held_out_shortcut_gain_vs_plain": held_gain_vs_plain,
        "held_out_shortcut_gain_vs_cf_aug": held_gain_vs_cf,
        "instability_drop_ok": bool(instability_drop_ok),
        "clean_drop_ok": bool(clean_drop_ok),
        "held_out_gain_ok": bool(held_out_gain_ok),
        "csa_promising": csa_promising,
    }


__all__ = [
    "TRAINING_MODES",
    "DataConfig",
    "TrainConfig",
    "CsaConfig",
    "PilotConfig",
    "EmbeddingSplit",
    "CsaDataset",
    "LinearHead",
    "build_dataset",
    "build_synthetic_dataset",
    "make_embedding_split",
    "train_head",
    "evaluate_split",
    "evaluate_mode",
    "run_modes",
    "compute_go_no_go",
    "clean_class_means",
    "nearest_class_mean_logits",
]
