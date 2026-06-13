from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from causal_reliability.discovery.candidate_factors import CandidateFactor


ApplyFn = Callable[[torch.Tensor, Optional[torch.Tensor], Optional[dict[str, Any]]], torch.Tensor]


@dataclass(frozen=True)
class CandidateIntervention:
    name: str
    factor_type: str
    apply: ApplyFn
    preserves_label_hint: bool
    support_score: float
    description: str
    affected_features: tuple[int | str, ...] = ()


def _object_mask(x: torch.Tensor) -> torch.Tensor:
    bg = x[:, :, :1, :1]
    return (x - bg).abs().mean(dim=1, keepdim=True) > 0.08


def _vector_apply(factor: CandidateFactor) -> ApplyFn:
    def apply(x: torch.Tensor, y: torch.Tensor | None = None, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        del y
        metadata = metadata or {}
        out = x.clone()
        if factor.candidate_type in {"individual_feature", "small_feature_group"}:
            if isinstance(factor.params.get("dim"), int):
                dims = (int(factor.params["dim"]),)
            else:
                dims = tuple(int(v) for v in factor.params.get("dims", ()))
            for dim in dims:
                out[:, dim] = torch.roll(out[:, dim], shifts=1, dims=0)
        elif factor.candidate_type == "random_linear_direction":
            direction = torch.tensor(factor.params["direction"], device=x.device, dtype=x.dtype)
            scale = float(metadata.get("random_direction_scale", 0.65)) * x.std(dim=0).mean().clamp_min(1e-6)
            out = out + scale * direction.view(1, -1)
        elif factor.candidate_type == "matched_magnitude_direction":
            dim = int(factor.params["dim"])
            delta = out[:, dim].std().clamp_min(1e-6)
            signs = torch.where(torch.arange(out.shape[0], device=out.device) % 2 == 0, 1.0, -1.0).to(out.dtype)
            out[:, dim] = out[:, dim] + signs * delta
        else:
            raise ValueError(f"unsupported vector candidate: {factor.candidate_type}")
        return out

    return apply


def _vision_apply(factor: CandidateFactor) -> ApplyFn:
    def apply(x: torch.Tensor, y: torch.Tensor | None = None, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        del y, metadata
        out = x.clone()
        mask = _object_mask(out)
        alt_colors = torch.tensor([[0.95, 0.20, 0.20], [0.15, 0.45, 0.95]], device=x.device, dtype=x.dtype)
        color = alt_colors[torch.arange(x.shape[0], device=x.device) % 2].view(x.shape[0], 3, 1, 1)
        if factor.candidate_type == "object_color":
            out = torch.where(mask, color.expand_as(out), out)
        elif factor.candidate_type == "background_color":
            out = torch.where(mask, out, (color * 0.55).expand_as(out))
        elif factor.candidate_type == "texture":
            h, w = out.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
            pattern = (0.65 + 0.25 * (((xx + yy) // 2) % 2).to(dtype=x.dtype)).view(1, 1, h, w)
            out = torch.where(mask, out * pattern, out)
        elif factor.candidate_type == "brightness":
            out = out + 0.12
        elif factor.candidate_type == "contrast":
            out = (out - 0.5) * 1.25 + 0.5
        elif factor.candidate_type == "small_translation":
            out = torch.roll(out, shifts=(1, 1), dims=(-2, -1))
        elif factor.candidate_type == "additive_noise":
            noise = torch.sin(torch.arange(out.numel(), device=x.device, dtype=x.dtype)).reshape_as(out) * 0.035
            out = out + noise
        elif factor.candidate_type == "blur":
            out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        elif factor.candidate_type == "shape_preserving_style":
            tint = torch.tensor([1.08, 0.96, 0.92], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            out = torch.where(mask, out * tint, out)
        else:
            raise ValueError(f"unsupported vision candidate: {factor.candidate_type}")
        return out.clamp(0, 1)

    return apply


def _paired_token(token: int, vocab: dict[str, int]) -> int:
    pairs = {
        vocab.get("always", -100): vocab.get("never", token),
        vocab.get("never", -100): vocab.get("always", token),
        vocab.get("low", -100): vocab.get("high", token),
        vocab.get("high", -100): vocab.get("low", token),
        vocab.get("less", -100): vocab.get("more", token),
        vocab.get("more", -100): vocab.get("less", token),
        vocab.get("clearly", -100): vocab.get("rarely", token),
        vocab.get("rarely", -100): vocab.get("clearly", token),
    }
    return int(pairs.get(token, token))


def _text_apply(factor: CandidateFactor) -> ApplyFn:
    def apply(x: torch.Tensor, y: torch.Tensor | None = None, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        del y
        metadata = metadata or {}
        vocab = metadata.get("vocab", {})
        out = x.clone()
        if factor.candidate_type == "individual_token":
            pos = int(factor.params["position"])
            alternatives = list(factor.params.get("alternatives") or [1, 2, 3, 4, 5, 6, 7, 8])
            replacement = torch.tensor(alternatives, device=x.device, dtype=x.dtype)
            out[:, pos] = replacement[torch.arange(x.shape[0], device=x.device) % len(replacement)]
        elif factor.candidate_type == "suspected_marker_token":
            token = int(factor.params["token"])
            out[out == token] = _paired_token(token, vocab)
        elif factor.candidate_type == "template_phrase":
            positions = tuple(int(v) for v in factor.params["positions"])
            for pos in positions:
                out[:, pos] = torch.tensor(_paired_token(int(out[0, pos]), vocab), device=x.device, dtype=x.dtype)
        elif factor.candidate_type == "surface_token_shuffle":
            positions = tuple(int(v) for v in factor.params["positions"])
            out[:, positions] = out[:, list(reversed(positions))]
        elif factor.candidate_type == "synonym_like_replacement":
            for pos in factor.params["positions"]:
                vals = [_paired_token(int(v), vocab) for v in out[:, int(pos)].tolist()]
                out[:, int(pos)] = torch.tensor(vals, device=x.device, dtype=x.dtype)
        elif factor.candidate_type == "neutral_filler_replacement":
            pos = int(factor.params["position"])
            out[:, pos] = torch.tensor(vocab.get("rarely", vocab.get("clearly", 7)), device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"unsupported text candidate: {factor.candidate_type}")
        return out

    return apply


def make_intervention(task_type: str, factor: CandidateFactor) -> CandidateIntervention:
    if task_type == "vector":
        support = 0.96 if factor.candidate_type != "random_linear_direction" else 0.82
        preserve = factor.candidate_type != "random_linear_direction"
        return CandidateIntervention(factor.candidate_id, factor.candidate_type, _vector_apply(factor), preserve, support, factor.description, factor.affected_features)
    if task_type == "vision":
        support_by_type = {
            "object_color": 0.98,
            "background_color": 0.96,
            "texture": 0.92,
            "brightness": 0.78,
            "contrast": 0.78,
            "small_translation": 0.88,
            "additive_noise": 0.68,
            "blur": 0.70,
            "shape_preserving_style": 0.88,
        }
        return CandidateIntervention(factor.candidate_id, factor.candidate_type, _vision_apply(factor), True, support_by_type.get(factor.candidate_type, 0.75), factor.description, factor.affected_features)
    if task_type == "text":
        support = 0.96 if factor.candidate_type in {"individual_token", "suspected_marker_token", "neutral_filler_replacement"} else 0.86
        return CandidateIntervention(factor.candidate_id, factor.candidate_type, _text_apply(factor), True, support, factor.description, factor.affected_features)
    raise ValueError(f"unknown task_type: {task_type}")
