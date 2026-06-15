---
name: visual-decoy-pilot-boundary
description: Priority-4 second-shortcut-family (non-text visual decoy) pilot is boundary evidence, not a positive result
metadata:
  type: project
---

The Priority-4 second-shortcut-family pilot (non-text visual decoy patch: central
causal shape + competing-class corner patch, no words) was run as a gated pilot on
OpenCLIP ViT-B-32 / laion2b_s34b_b79k at n_per_condition=64. It is **boundary
evidence**, NOT a positive headline result.

Outcome: 7/8 strict gates pass (clean acc 1.0, oracle repair 0.969, CIC top-1 minus
matched-random 0.297, clean-safe drop 0.0, no scorer leakage). The single failing
gate is `misleading_original_le_0_40`: a single non-text competing-shape decoy only
fools zero-shot CLIP to ~0.578 (floors ~0.50–0.56 across ~11 stimulus probes) because
CLIP anchors on the dominant central shape. The shortcut is real but too weak.

Scaling to n=128 / multiple models is NOT recommended — the binding failure is
stimulus strength, which more samples cannot fix.

Code/artifacts (all isolated; did NOT touch text-overlay headline or scale-and-multi-model audit):
- causal_reliability/data/clip_visual_decoy_shortcuts.py
- causal_reliability/experiments/run_visual_decoy_clip_pilot.py
- configs/visual_decoy_pilot.yaml, tests/test_visual_decoy_pilot.py
- results/visual_decoy_pilot/
- additive helper proposal_from_bbox in [[region_proposals]] (discovery/region_proposals.py)
