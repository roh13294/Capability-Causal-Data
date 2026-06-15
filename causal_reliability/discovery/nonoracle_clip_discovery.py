from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from causal_reliability.discovery.cic_region_scoring import RegionScore, score_region_candidates
from causal_reliability.discovery.region_proposals import RegionProposal, generate_region_proposals


def discover_clip_shortcut_regions(
    image: Image.Image | np.ndarray,
    predict_fn: Callable[[list[Image.Image]], np.ndarray],
    class_prompts: list[str],
    seed: int = 0,
    max_candidates: int = 96,
) -> tuple[list[RegionProposal], list[RegionScore], np.ndarray]:
    """Generate and rank candidate shortcut regions without oracle metadata.

    `class_prompts` is accepted to make the discovery input contract explicit for
    CLIP callers. The region scorer uses pixels, proposals, and model
    probabilities, including the model's original top prediction as an
    inference-time signal; it does not receive labels or overlay metadata.
    """

    pil = Image.fromarray((np.asarray(image).clip(0, 1) * 255).astype(np.uint8)).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
    _ = class_prompts
    proposals = generate_region_proposals(pil, seed=seed, max_candidates=max_candidates)
    scores, original_probs = score_region_candidates(pil, proposals, predict_fn)
    return proposals, scores, original_probs
