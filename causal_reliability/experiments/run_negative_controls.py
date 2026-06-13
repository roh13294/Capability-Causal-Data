from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.experiments.stress_utils import certificate_frame, failure_metrics, shift_risk_summary, train_for_bundle, write_json
from causal_reliability.training.eval import collect_logits
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def _replace_shortcut(x: torch.Tensor, source: torch.Tensor, n_counterfactuals: int) -> torch.Tensor:
    out = x.unsqueeze(1).repeat(1, n_counterfactuals, 1)
    out[:, :, 1] = source.view(-1, 1)
    return out


def irrelevant_counterfactuals(x: torch.Tensor, n_counterfactuals: int = 4) -> torch.Tensor:
    out = x.unsqueeze(1).repeat(1, n_counterfactuals, 1)
    out[:, :, -1] = torch.linspace(-1.0, 1.0, n_counterfactuals, device=x.device, dtype=x.dtype).view(1, -1)
    return out


def random_intervention_direction(x: torch.Tensor, n_counterfactuals: int = 4) -> torch.Tensor:
    out = x.unsqueeze(1).repeat(1, n_counterfactuals, 1)
    out[:, :, 2] = x[:, 2].view(-1, 1) + torch.linspace(-1.4, 1.4, n_counterfactuals, device=x.device, dtype=x.dtype).view(1, -1)
    return out


def shuffled_index_by_label(labels: torch.Tensor) -> torch.Tensor:
    perm = torch.empty_like(labels)
    for label in labels.unique():
        idx = (labels == label).nonzero(as_tuple=False).flatten()
        perm[idx] = idx[torch.randperm(len(idx))]
    return perm


def shuffled_index_by_shortcut(shortcuts: torch.Tensor) -> torch.Tensor:
    perm = torch.empty_like(shortcuts)
    for shortcut in shortcuts.unique():
        idx = (shortcuts == shortcut).nonzero(as_tuple=False).flatten()
        perm[idx] = idx[torch.randperm(len(idx))]
    return perm


def matched_confidence_index(confidence: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(confidence)
    perm = torch.empty_like(order)
    if len(order) == 1:
        perm[order] = order
        return perm
    perm[order[:-1]] = order[1:]
    perm[order[-1]] = order[-2]
    return perm


def _indexed_counterfactual_factory(bundle, perm: torch.Tensor, n_counterfactuals: int):
    source = bundle.shifted_test.tensors[0][perm, 1]
    cursor = {"i": 0}

    def make_cf(x: torch.Tensor) -> torch.Tensor:
        start = cursor["i"]
        end = start + x.shape[0]
        cursor["i"] = end
        return _replace_shortcut(x, source[start:end].to(x.device), n_counterfactuals)

    return make_cf


def _random_label_bundle(cfg: dict[str, Any]):
    bundle = make_vector_task(**cfg)
    train = list(bundle.train.tensors)
    shifted = list(bundle.shifted_test.tensors)
    train[1] = torch.randint(0, 2, train[1].shape)
    shifted[1] = torch.randint(0, 2, shifted[1].shape)
    from causal_reliability.data.splits import DatasetBundle, tensor_dataset

    return DatasetBundle(tensor_dataset(*train), bundle.id_test, tensor_dataset(*shifted), input_shape=bundle.input_shape, task_type=bundle.task_type)


def _confidence_for_shifted(model: torch.nn.Module, bundle, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(bundle.shifted_test, batch_size=int(cfg.get("batch_size", 64)))
    logits, _labels = collect_logits(model, loader, device)
    return torch.softmax(logits, dim=1).max(dim=1).values.cpu()


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "negative_controls")
    base_data = dict(cfg.get("data", {}))
    base_data.setdefault("shift_mode", "partial_in_support_flip")
    base_data.setdefault("partial_flip_fraction", 0.7)
    metric_rows = []
    failure_rows = []
    controls = [
        "true_counterfactual",
        "random_labels",
        "irrelevant_counterfactual",
        "shuffled_any",
        "shuffled_within_class",
        "shuffled_same_shortcut",
        "shuffled_matched_confidence",
        "random_intervention_direction",
    ]
    for i, name in enumerate(controls):
        task_cfg = dict(cfg)
        task_cfg["seed"] = int(cfg.get("seed", 0)) + i
        task_cfg["data"] = base_data
        bundle = _random_label_bundle(base_data) if name == "random_labels" else make_vector_task(**base_data)
        model, id_metrics, shifted_metrics, device = train_for_bundle(bundle, task_cfg, int(task_cfg["seed"]))
        n_cf = int(task_cfg.get("n_counterfactuals", 4))
        x, y, shortcut, _causal = bundle.shifted_test.tensors
        make_cf = None
        if name == "irrelevant_counterfactual":
            make_cf = lambda batch, n=n_cf: irrelevant_counterfactuals(batch, n)
        elif name == "random_intervention_direction":
            make_cf = lambda batch, n=n_cf: random_intervention_direction(batch, n)
        elif name == "shuffled_any":
            make_cf = _indexed_counterfactual_factory(bundle, torch.randperm(len(x)), n_cf)
        elif name == "shuffled_within_class":
            make_cf = _indexed_counterfactual_factory(bundle, shuffled_index_by_label(y), n_cf)
        elif name == "shuffled_same_shortcut":
            make_cf = _indexed_counterfactual_factory(bundle, shuffled_index_by_shortcut(shortcut), n_cf)
        elif name == "shuffled_matched_confidence":
            make_cf = _indexed_counterfactual_factory(bundle, matched_confidence_index(_confidence_for_shifted(model, bundle, task_cfg, device)), n_cf)
        cert_df = certificate_frame(model, bundle, task_cfg, device, make_cf=make_cf)
        cert_df.insert(0, "control", name)
        cert_df.to_csv(out_dir / f"{name}_certificates.csv", index=False)
        failure = failure_metrics(cert_df)
        failure.insert(0, "control", name)
        failure_rows.append(failure)
        metric_rows.append(
            {
                "control": name,
                "negative_control_type": name if name != "true_counterfactual" else "",
                "id_accuracy": id_metrics["accuracy"],
                "shifted_accuracy": shifted_metrics["accuracy"],
                "failure_rate": float(cert_df["failure"].mean()),
                **shift_risk_summary(cert_df),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    failures = pd.concat(failure_rows, ignore_index=True)
    metrics.to_csv(out_dir / "negative_control_metrics.csv", index=False)
    failures.to_csv(out_dir / "negative_control_failure_prediction.csv", index=False)
    write_json(out_dir / "negative_control_summary.json", {"controls": metric_rows})
    return {"out_dir": str(out_dir), "controls": metric_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/negative_controls.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
