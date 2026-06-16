# COCO-Text CIC Triage (pre-flight)

Lightweight triage over the curated COCO-Text x COCO-objects metadata sample. 
Oracle text-box masking uses **global, label-free operators only** (gray fill, 
expanded gray fill 1.25, Gaussian blur, expanded blur 1.25). **No open-proposal 
CIC was run.** Text/object boxes are eval-only geometry for the oracle upper bound.

Backend: `open_clip`. Model: `ViT-B-32`. 
Real pretrained loaded: `True`. Fake backend: `False`.
Data: loaded 500 local images.

**Result: COCO-Text sample is READY for a full proposal-CIC run.**
`coco_text_ready_for_full_cic = True`.
All gate conditions met.

**Recommendation:** Proceed to the full proposal-CIC run on this metadata sample.

## Key numbers

- Metadata rows loaded: 500
- Real pretrained model loaded: True
- Fake backend: False
- Original CLIP accuracy (alias-aware): 0.76
- High-confidence failure rate: 0.118
- Directional verified failures: 57
- Strict oracle-repairable failures: 39
- Oracle top-5/pairwise recoveries (over directional): 56
- Oracle strict top-1 rate (over directional): 0.3157894736842105
- Oracle strict top-3 rate (over directional): 0.8596491228070176
- Oracle strict top-5 rate (over directional): 0.9649122807017544
- Oracle target-probability improvement rate (over failures): 0.49166666666666664
- Oracle text-distractor decrease rate (over failures): 0.5916666666666667
- Clean-subset examples: 219

## Gate status

- `coco_text_ready_for_full_cic`: True
- Thresholds: directional >= 50, strict >= 30 OR top5/pairwise >= 50, clean >= 30
- Failed gate reasons: none

## Categories

- Strongest (most strict-repairable): car, truck, laptop, pizza, cup
- Weakest (images but no directional failures): apple, bicycle, bird, boat, bus

## Examples to inspect

- 7:truck (op=expanded_gray_fill_1.25)
- 29:pizza (op=gray_fill)
- 56:cow (op=gaussian_blur)
- 61:car (op=gaussian_blur)
- 69:truck (op=expanded_gray_fill_1.25)
- 76:orange (op=gaussian_blur)
- 77:car (op=expanded_blur_1.25)
- 88:car (op=expanded_blur_1.25)

## Scope guard

- This is a triage pass. Open-proposal CIC was **not** run.
- Oracle operators are global and label-free; text boxes are eval-only geometry.
- Writes only under this output subdirectory; no final-report metric was touched.

## Per-category summary

| category   | n_images | n_directional | n_strict | strict_rate |
| ---------- | -------- | ------------- | -------- | ----------- |
| car        | 37       | 13            | 9        | 0.2432      |
| truck      | 41       | 10            | 6        | 0.1463      |
| laptop     | 21       | 5             | 4        | 0.1905      |
| pizza      | 34       | 3             | 2        | 0.05882     |
| cup        | 11       | 3             | 1        | 0.09091     |
| bottle     | 8        | 2             | 2        | 0.25        |
| sandwich   | 14       | 2             | 2        | 0.1429      |
| train      | 21       | 2             | 2        | 0.09524     |
| umbrella   | 27       | 2             | 2        | 0.07407     |
| cat        | 14       | 2             | 1        | 0.07143     |
| motorcycle | 28       | 2             | 1        | 0.03571     |
| backpack   | 8        | 2             | 0        | 0           |
| airplane   | 24       | 1             | 1        | 0.04167     |
| banana     | 12       | 1             | 1        | 0.08333     |
| cell phone | 7        | 1             | 1        | 0.1429      |
| clock      | 14       | 1             | 1        | 0.07143     |
| cow        | 6        | 1             | 1        | 0.1667      |
| orange     | 8        | 1             | 1        | 0.125       |
| vase       | 5        | 1             | 1        | 0.2         |
| broccoli   | 5        | 1             | 0        | 0           |
| cake       | 21       | 1             | 0        | 0           |
| apple      | 3        | 0             | 0        | 0           |
| bear       | 1        | 0             | 0        | 0           |
| bicycle    | 7        | 0             | 0        | 0           |
| bird       | 7        | 0             | 0        | 0           |
| boat       | 4        | 0             | 0        | 0           |
| bus        | 46       | 0             | 0        | 0           |
| carrot     | 4        | 0             | 0        | 0           |
| dog        | 13       | 0             | 0        | 0           |
| donut      | 2        | 0             | 0        | 0           |
| elephant   | 5        | 0             | 0        | 0           |
| giraffe    | 4        | 0             | 0        | 0           |
| horse      | 10       | 0             | 0        | 0           |
| hot dog    | 5        | 0             | 0        | 0           |
| suitcase   | 5        | 0             | 0        | 0           |
| teddy bear | 16       | 0             | 0        | 0           |
| zebra      | 2        | 0             | 0        | 0           |