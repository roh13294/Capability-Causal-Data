---
name: predictive-cic-gate
description: The predictive CIC reliability gate experiment, its support status, and its key honest caveat
metadata:
  type: project
---

`results/predictive_cic_gate/` holds a label-free predictive reliability gate
(added 2026-06-16) that predicts whether a CIC repair should be trusted before the
true label is seen. Code: `causal_reliability/analysis/predictive_cic_gate.py`,
`causal_reliability/experiments/run_predictive_cic_gate.py`,
`configs/predictive_cic_gate.yaml`. Theory: `causal_reliability/theory/predictive_certificate.py`
+ docs/theory.md "Appendix P".

`predictive_gate_supported = True` (2635 examples, 8 benchmarks, pooled
leave-one-benchmark-out AUROC ≈0.79, top-25% accepted-repair precision ≈0.97).

**Why / non-obvious caveat:** the pooled LOBO number is optimistic because the
scale-audit runs are near-duplicates of hard_multidecoy, so each fold still trains
on a same-family benchmark. **Pure controlled→natural transfer is at chance
(AUROC ≈0.50)**, and the gate is *below chance* on the COCO-Text strict (0.35) and
directional (0.30) subsets. These are recorded in the run's `caveats` field — do
not present the gate as transferring to hard natural failures.

**How to apply:** it is a practical predictive layer, NOT a universal theorem, and
it writes only under `results/predictive_cic_gate/`. It never touches
`results/final_report/` or existing support gates. See [[]] none.
