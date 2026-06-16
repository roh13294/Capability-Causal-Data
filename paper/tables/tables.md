# Paper Tables (markdown mirror)

All numbers from current repository artifacts.

## Table 1 — Hard multi-decoy held-out benchmark (OpenCLIP ViT-B-32, n=32/condition)
Source: `results/final_report/final_key_numbers.json`

| Condition / method | Accuracy | 95% CI |
|---|---|---|
| No-overlay | 1.000 | [0.893, 1.000] |
| Aligned overlay | 1.000 | [0.893, 1.000] |
| Misleading (original) | 0.250 | [0.133, 0.421] |
| Oracle harmful-text repair (upper bound) | 1.000 | [0.893, 1.000] |
| Matched random text-region repair | 0.331 | ±0.010 (seed) |
| **CIC top-1 repair** | **0.750** | [0.579, 0.867] |
| **CIC top-3 repair** | **0.750** | [0.579, 0.867] |
| CIC clean-safe repair | 0.750 | [0.579, 0.867] |
| CIC top-1 − matched random (gap) | 0.419 | [0.409, 0.430] |
| Clean-safe clean drop | 0.010 | — |
| Top-1 localization IoU ≥ 0.3 | 0.594 | [0.423, 0.745] |
| Top-1 localization IoU ≥ 0.5 | 0.063 | — |

## Table 2 — Failure-conditioned repair (n=50 verified failures; original = 0 by construction)
Source: `results/hard_multidecoy_failure_conditioned/`

| Method | Repaired accuracy | 95% CI |
|---|---|---|
| Original (by construction) | 0.000 | [0.000, 0.071] |
| Oracle harmful-text repair (upper bound) | 1.000 | [0.929, 1.000] |
| **CIC top-1 repair** | **0.960** | [0.865, 0.989] |
| **CIC top-3 repair** | **0.980** | [0.893, 0.996] |
| CIC clean-safe repair | 0.940 | [0.838, 0.979] |
| Highest-textness repair | 0.820 | [0.692, 0.902] |
| Largest-text repair | 0.260 | [0.159, 0.396] |
| Matched random text-region repair | 0.112 | ±0.015 |
| Random augmentation consensus | 0.020 | [0.004, 0.105] |
| Random non-text patch repair | 0.002 | [0.001, 0.007] |

## Table 3 — Full benchmark-resampling audit (3 resampled instances, distinct hashes)
Source: `results/hard_multidecoy_clip_repair/full_benchmark_resampling_*`

| Quantity | Mean | Min | Max |
|---|---|---|---|
| Original misleading accuracy | 0.260 | 0.250 | 0.281 |
| CIC top-1 repair | 0.750 | 0.719 | 0.813 |
| Matched random text repair | 0.289 | 0.268 | 0.329 |
| CIC − random gap | 0.461 | 0.390 | 0.545 |
| Clean-safe clean drop | 0.014 | — | — |

`full_benchmark_resampling_stability_supported = true`

## Table 4 — Theory validation
Sources: `results/embedding_additivity/`, `results/per_input_class_balance/`

Global embedding additivity (supported_for_text = **false**):

| Metric | Value |
|---|---|
| within-shortcut cosine | 0.765 |
| shuffled-label baseline cosine | 0.634 |
| within-object cosine (> within-shortcut) | 0.855 |
| neutralization damage / shortcut-effect ratio | 0.917 |
| logit-channel consistency MAE | 6e-15 |

Per-input class-balance (supported_for_text = **true**, ε_B = 3.0):

| Condition | Median residual | Repair acc. | Margin-cond. rate |
|---|---|---|---|
| Oracle | 2.464 | 1.000 | 0.719 |
| CIC top-1 | 3.704 | 0.781 | 0.344 |
| CIC top-3 (consensus) | 3.506 | 0.781 | 0.344 |
| Matched random text | 5.220 | 0.406 | 0.125 |

## Table 5 — Negative / boundary results
Sources: `results/cross_shortcut_generalization/`, `results/embedding_additivity/`

| Result | Value | Status |
|---|---|---|
| Cross-shortcut natural misleading accuracy | 0.750 | shortcut weak |
| Cross-shortcut failure inclusion rate | 0.010 (4/384) | too few failures |
| Cross-shortcut frozen CIC top-1 (failure subset) | 0.500 | below threshold |
| Cross-shortcut headline eligible | false | **no transfer** |
| Global additivity supported (text) | false | object-entangled |
| Global additivity supported (watermark) | false | weak channel |
| Exact localization IoU ≥ 0.5 (main) | 0.063 | coarse only |

## Table 6 — Reproducibility checklist

| Item | Status |
|---|---|
| Real pretrained OpenCLIP loaded (no fake backend for headline) | yes |
| Non-oracle scorer excludes label/bbox/type/correctness | yes |
| Headline gated on `headline_eligible` | yes |
| Full benchmark resampling with distinct image/metadata hashes | yes (3 seeds) |
| Fixed-benchmark determinism check separated from resampling | yes |
| Global-additivity theory claim gated on validation metric | yes (false) |
| Per-input class-balance gate | yes (true for text) |
| Cross-shortcut transfer reported honestly when failing | yes (not eligible) |
| Spatial-resolution & causal-intervention audit (diagnostic; not exact localization) | yes (pooled n=210; coarse-intervention framing) |
| Test suite | 382/382 pass |
| Negative controls | 24/24 pass |
| Human label-preservation study (3 annotators, 100 pairs) | majority-vote preserved 96/100, recognizable 97/100; Fleiss' kappa up to 1.000 |
