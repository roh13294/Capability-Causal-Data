# COCO-Text Automated Proposal Sweep — Apples-to-Apples Summary

This is **automated finite-candidate proposal generation**: CIC still scores a 
finite candidate set; only the *source* of the candidate boxes changes (manual 
open-proposal generator vs. automatic grid / edge-component / saliency generators). 
Every family is scored and evaluated with identical code, and compared **only** 
against the apples-to-apples finalized baseline `existing_cic_baseline_a2a` that 
exactly reproduces the finalized `cic_top1_repair_excl_ocr` recipe.

## Non-claims (explicit, bounded language)

- This is **not** open-world shortcut discovery.
- This is **not** universal repair or general robustness.
- This is **not** deployment validation or clinical validation.
- This is **not** a replacement for the finalized STS report.

## 1. Reconciliation result

- Finalized strict CIC repair (headline): **0.538** (matched random 0.205).
- First pilot `existing_cic_baseline` strict CIC repair: **0.410**.
- Pilot baseline directly comparable to finalized result: **False** (different candidate cap 14 vs 48, different grid scales, and OCR geometry disabled — see `coco_reconciliation.md`).
- Apples-to-apples baseline reproduces the finalized strict number exactly: **True**.

- Backend: `open_clip` (ViT-B-32); a2a candidate cap 48, grid scales [0.18, 0.3, 0.45].
- Families evaluated: existing_cic_baseline_a2a, grid_boxes, edge_component_boxes, saliency_boxes, sam_boxes.

## 1b. SAM (Segment Anything) status

- SAM loaded successfully: **True** (device `cpu`); run state: **complete**.
- Settings: model_type=`vit_b`, checkpoint=`models/sam/sam_vit_b_01ec64.pth`, fast=True, points_per_side=8, crop_n_layers=0, max_side=512, pred_iou_thresh=0.86, stability_score_thresh=0.9, max_proposals=48, area_frac∈[0.002, 0.8], min_side=8, dedupe_iou=0.7.
- Runtime: 182.001 s SAM generation (total run 555.490 s); images completed: 39; cache hits/misses: 3/36; timeout budget 1800.000 s.
- SAM masks → XYWH→XYXY boxes, downscaled (max_side) for speed, filtered by side/area, IoU-deduplicated, top-K by predicted_iou / stability_score / area, and cached per image. This stays **automated finite-candidate proposal generation**, not open-world shortcut discovery.

## 2. Full strict / directional / all-500 table

### Subset `strict` (n=39, original alias accuracy=0.000)

| family | CIC repair | random repair | gap | tgt-prob↑ | distr↓ | sel text-ovl | sel obj-ovl | cov@.1 | cov@.3 | med rank | sel area | wins/losses vs a2a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| existing_cic_baseline_a2a | 0.538 | 0.205 | 0.333 | 0.718 | 0.949 | 0.128 | 0.333 | 0.564 | 0.308 | 5.500 | 0.060 | — (baseline) |
| grid_boxes | 0.462 | 0.359 | 0.103 | 0.795 | 0.897 | 0.205 | 0.179 | 0.641 | 0.333 | 5.000 | 0.032 | 3/6 (net -3) |
| edge_component_boxes | 0.487 | 0.333 | 0.154 | 0.769 | 0.949 | 0.128 | 0.231 | 0.718 | 0.487 | 7.000 | 0.030 | 4/6 (net -2) |
| saliency_boxes | 0.513 | 0.385 | 0.128 | 0.744 | 0.872 | 0.205 | 0.231 | 0.564 | 0.308 | 3.000 | 0.062 | 4/5 (net -1) |
| sam_boxes | 0.462 | 0.385 | 0.077 | 0.744 | 0.923 | 0.128 | 0.308 | 0.590 | 0.385 | 4.000 | 0.058 | 3/6 (net -3) |

## 3. Do automated proposals genuinely improve over the finalized baseline?

- `grid_boxes`: strict repair gain -0.077, directional repair drop n/a, text-overlap gain 0.077, coverage gain 0.077 → **promotable=False** (A=False, B=False, C=False)
- `edge_component_boxes`: strict repair gain -0.051, directional repair drop n/a, text-overlap gain 0.000, coverage gain 0.154 → **promotable=False** (A=False, B=False, C=False)
- `saliency_boxes`: strict repair gain -0.026, directional repair drop n/a, text-overlap gain 0.077, coverage gain 0.000 → **promotable=False** (A=False, B=False, C=False)
- `sam_boxes`: strict repair gain -0.077, directional repair drop n/a, text-overlap gain 0.000, coverage gain 0.026 → **promotable=False** (A=False, B=False, C=False)

### SAM-specific promotion rule

- `sam_boxes`: strict repair gain -0.077, directional repair gain n/a, text-overlap gain 0.000, coverage gain 0.026 → **sam_promotable=False** (A=False, B=False, C=False, D=False)

## 4. Promotion verdict

- **auto_proposal_promotable = False**
- **sam_promotable = False**
- No automated proposal family beats the apples-to-apples finalized baseline by a pre-registered margin. Preserved honestly as a negative/diagnostic result: automated proposals are competitive but **not** a promotable improvement over the finalized hand-designed candidate set.

## 5. Explicit non-claims

- This is **not** open-world discovery — it is automated finite-candidate proposal generation.
- This is **not** universal repair.
- This is **not** deployment validation.
- This is **not** a replacement for the finalized STS report; the finalized COCO-Text and STS numbers and support gates are unchanged.

