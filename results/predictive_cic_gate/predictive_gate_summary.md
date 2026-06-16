# Predictive CIC Reliability Gate

**Practical predictive reliability layer on top of the CIC framework — NOT a new universal theorem.**

This gate predicts, from inference-time observable features only (no true label, no
target label, no correctness, no oracle repair success, no support-subset membership,
no ground-truth box overlap), whether a CIC repair should be trusted *before* the true
label is revealed.

## Dataset
- Examples in predictive dataset: **2635** (2635 with primary label)
- Benchmarks included: coco_text_full, hard_multidecoy, natural_text_open, natural_text_verified, scale_audit::RN50__openai, scale_audit::ViT-B-16__laion2b_s34b_b88k, scale_audit::ViT-B-32__openai, semantic_decoy
- Group sizes: {'controlled': 2048, 'natural': 587}
- Primary label: `label_repair_success` (base success rate 0.785)

## Label-free / leakage guards
- Features label-free: **True**
- No-oracle-leakage check passed: **True**
- Leakage reasons: none
- Real-backend evidence (provenance): **True**

## Cross-benchmark validation
- Leave-one-benchmark-out pooled AUROC (`logistic_regression`): **0.789** (AUPRC 0.929)
  - held-out `coco_text_full`: AUROC 0.699, AUPRC 0.798, base 0.714, n=500 
  - held-out `hard_multidecoy`: AUROC 0.942, AUPRC 0.990, base 0.852, n=128 
  - held-out `natural_text_open`: AUROC 0.395, AUPRC 0.176, base 0.200, n=50 
  - held-out `natural_text_verified`: AUROC 0.616, AUPRC 0.332, base 0.216, n=37 
  - held-out `scale_audit::RN50__openai`: AUROC 0.775, AUPRC 0.935, base 0.871, n=512 
  - held-out `scale_audit::ViT-B-16__laion2b_s34b_b88k`: AUROC 0.925, AUPRC 0.990, base 0.895, n=512 
  - held-out `scale_audit::ViT-B-32__openai`: AUROC 0.846, AUPRC 0.965, base 0.834, n=512 
  - held-out `semantic_decoy`: AUROC 0.915, AUPRC 0.949, base 0.659, n=384 
- train_controlled_test_coco (logistic): AUROC 0.509, n=500
- train_controlled_test_natural (logistic): AUROC 0.503, n=587

## Reporting subsets (LOBO out-of-fold scores)
| subset | n | base rate | AUROC | AUPRC | Brier |
|---|---|---|---|---|---|
| controlled | 2048 | 0.827 | 0.832 | 0.955 | 0.166 |
| coco_strict | 39 | 0.538 | 0.347 | 0.439 | 0.387 |
| coco_directional | 57 | 0.439 | 0.296 | 0.333 | 0.419 |
| coco_all | 500 | 0.714 | 0.699 | 0.798 | 0.168 |
| natural_all | 587 | 0.639 | 0.661 | 0.723 | 0.219 |
| overall_pooled | 2635 | 0.785 | 0.789 | 0.929 | 0.178 |

## COCO-Text held-out (reported separately)
- COCO-Text strict subset AUROC: **0.347**
- COCO-Text directional subset AUROC: **0.296**
- COCO-Text all AUROC: **0.699**

## Accept top-X% most trustworthy repairs
If the gate accepts only the top-X% most trustworthy CIC repairs, accepted-repair precision is:
- top_10pct_precision: 0.989
- top_20pct_precision: 0.985
- top_25pct_precision: 0.968
- top_30pct_precision: 0.968
- top_50pct_precision: 0.930

## Best simple interpretable rule
- Rule: `accept CIC repair iff feat_prediction_changed <= 0`
- LOBO out-of-fold AUROC of this single feature: 0.749

## Best predictive features (univariate, |AUROC-0.5| ranked)
- `feat_prediction_changed`: univariate AUROC 0.203 (coverage 0.967)
- `feat_repaired_confidence`: univariate AUROC 0.737 (coverage 0.777)
- `feat_repaired_margin`: univariate AUROC 0.737 (coverage 0.777)
- `feat_stability_gain`: univariate AUROC 0.266 (coverage 0.840)
- `feat_repaired_entropy`: univariate AUROC 0.269 (coverage 0.777)
- `feat_topk_repair_agreement`: univariate AUROC 0.717 (coverage 0.876)

## Conservative support flag
- `predictive_gate_supported`: **True**
- LOBO AUROC: 0.789 (floor 0.75)
- Best high-confidence operating point: coverage 0.950, precision 0.823 (floors: coverage 0.25, precision 0.80)
- Reasons gate is not supported: none (all criteria met)

## Honest caveats
- weak transfer: held-out predictive AUROC on 'coco_strict' is 0.347 (<=0.55); the gate does not reliably rank repairs there
- weak transfer: held-out predictive AUROC on 'coco_directional' is 0.296 (<=0.55); the gate does not reliably rank repairs there
- weak transfer: held-out predictive AUROC on 'natural_text_open' is 0.395 (<=0.55); the gate does not reliably rank repairs there
- cross-benchmark protocol 'train_controlled_test_coco' AUROC is 0.509 (near chance); pure controlled->natural transfer is weak. The pooled LOBO AUROC is higher because each fold still trains on some same-family benchmarks (the scale-audit runs are near-duplicates of hard_multidecoy).
- cross-benchmark protocol 'train_controlled_test_natural' AUROC is 0.503 (near chance); pure controlled->natural transfer is weak. The pooled LOBO AUROC is higher because each fold still trains on some same-family benchmarks (the scale-audit runs are near-duplicates of hard_multidecoy).

## Integrity
- Final report unchanged: **True** (sha256 aec85f2cc358528ddbb3f33c627510f5c7f57dd571c90a53c4a09e873646da8f)
- This experiment wrote only under `results/predictive_cic_gate/`.

## Proposition
See `docs/theory.md` (Predictive CIC certificate) and
`causal_reliability/theory/predictive_certificate.py`:

> If the repaired prediction margin exceeds an empirically calibrated
> residual-instability bound, then the repaired prediction is stable under the
> calibrated perturbation class. This does not prove universal correctness; it
> gives a label-free abstention rule for deciding when CIC repair is reliable.

Notes: ['hard_multidecoy: 128 examples', 'scale_model_audit: 1536 examples', 'semantic_decoy: 384 examples', 'coco_text_full: 500 examples', 'natural_text_open: 50 examples', 'natural_text_verified: 37 examples']