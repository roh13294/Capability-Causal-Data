from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateFactor:
    candidate_id: str
    candidate_type: str
    affected_features: tuple[int | str, ...]
    description: str
    params: dict[str, Any] = field(default_factory=dict)


def synthetic_candidate_factors(n_features: int, seed: int = 0, max_groups: int = 6, n_random: int = 4) -> list[CandidateFactor]:
    factors: list[CandidateFactor] = []
    for dim in range(n_features):
        factors.append(
            CandidateFactor(
                candidate_id=f"feature_dim_{dim}",
                candidate_type="individual_feature",
                affected_features=(dim,),
                description=f"Perturb individual feature dimension {dim}.",
                params={"dim": dim},
            )
        )
    for i, group in enumerate(combinations(range(n_features), 2)):
        if i >= max_groups:
            break
        factors.append(
            CandidateFactor(
                candidate_id="feature_group_" + "_".join(str(v) for v in group),
                candidate_type="small_feature_group",
                affected_features=tuple(group),
                description=f"Perturb small feature group {group}.",
                params={"dims": tuple(group)},
            )
        )
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        direction = rng.normal(size=n_features)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        factors.append(
            CandidateFactor(
                candidate_id=f"random_direction_{i}",
                candidate_type="random_linear_direction",
                affected_features=tuple(range(n_features)),
                description="Perturb along a deterministic random linear direction.",
                params={"direction": direction.astype(float).tolist()},
            )
        )
    for dim in range(n_features):
        factors.append(
            CandidateFactor(
                candidate_id=f"matched_magnitude_dim_{dim}",
                candidate_type="matched_magnitude_direction",
                affected_features=(dim,),
                description=f"Move feature dimension {dim} by a matched empirical magnitude.",
                params={"dim": dim},
            )
        )
    return factors


def vision_candidate_factors() -> list[CandidateFactor]:
    specs = [
        ("object_color", "object_color", ("object",), "Change object color while preserving shape."),
        ("background_color", "background_color", ("background",), "Change background color while preserving object geometry."),
        ("texture", "texture", ("object_texture",), "Apply a shape-preserving object texture perturbation."),
        ("brightness", "brightness", ("global_brightness",), "Adjust image brightness."),
        ("contrast", "contrast", ("global_contrast",), "Adjust image contrast."),
        ("translation", "small_translation", ("position",), "Translate the image by a small amount."),
        ("additive_noise", "additive_noise", ("pixels",), "Add mild deterministic pixel noise."),
        ("blur", "blur", ("pixels",), "Apply a small blur."),
        ("style", "shape_preserving_style", ("style",), "Apply a shape-preserving style/color-temperature perturbation."),
    ]
    return [CandidateFactor(cid, ctype, features, desc) for cid, ctype, features, desc in specs]


def text_candidate_factors(vocab: dict[str, int] | None = None, seq_len: int = 6) -> list[CandidateFactor]:
    vocab = vocab or {}
    inv = {v: k for k, v in vocab.items()}
    factors: list[CandidateFactor] = []
    seen_tokens = tuple(sorted(v for v in vocab.values() if v != vocab.get("<pad>", -1)))
    for pos in range(seq_len):
        token_names = ", ".join(inv.get(v, str(v)) for v in seen_tokens[:4])
        factors.append(
            CandidateFactor(
                candidate_id=f"token_position_{pos}",
                candidate_type="individual_token",
                affected_features=(pos,),
                description=f"Replace token position {pos} with seen alternatives ({token_names}).",
                params={"position": pos, "alternatives": seen_tokens},
            )
        )
    for token in ("always", "never"):
        if token in vocab:
            factors.append(
                CandidateFactor(
                    candidate_id=f"suspected_marker_{token}",
                    candidate_type="suspected_marker_token",
                    affected_features=(token,),
                    description=f"Replace occurrences of marker-like token '{token}' with its paired alternative.",
                    params={"token": vocab[token]},
                )
            )
    factors.extend(
        [
            CandidateFactor("template_phrase_surface", "template_phrase", (2, 3), "Replace the shortcut/filler template phrase.", params={"positions": (2, 3)}),
            CandidateFactor("shuffle_surface_tokens", "surface_token_shuffle", (2, 3, 4, 5), "Shuffle non-leading surface tokens deterministically.", params={"positions": (2, 3, 4, 5)}),
            CandidateFactor("synonym_like_leading_terms", "synonym_like_replacement", (0, 1), "Swap seen low/high and less/more alternatives.", params={"positions": (0, 1)}),
            CandidateFactor("neutral_filler", "neutral_filler_replacement", (3,), "Replace filler tokens with neutral seen alternatives.", params={"position": 3}),
        ]
    )
    return factors


def generate_candidate_factors(task_type: str, **kwargs: Any) -> list[CandidateFactor]:
    if task_type == "vector":
        return synthetic_candidate_factors(int(kwargs.get("n_features", 4)), int(kwargs.get("seed", 0)))
    if task_type == "vision":
        return vision_candidate_factors()
    if task_type == "text":
        return text_candidate_factors(kwargs.get("vocab"), int(kwargs.get("seq_len", 6)))
    raise ValueError(f"unknown task_type: {task_type}")
