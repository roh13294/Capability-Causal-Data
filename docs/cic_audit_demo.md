# CIC Audit Demo

This walkthrough shows the practitioner-facing audit workflow for settings where candidate shortcut interventions can be specified. It is not a turnkey solution for arbitrary models or unknown shortcuts.

## Install

```bash
pip install -e .
```

## Run The Demo

```bash
python3 -m causal_reliability.audit.run_cic_audit --config configs/example_cic_audit.yaml
```

## Workflow

1. Load examples with `example_id`, `text`, and optional `label`.
2. Provide a `model_predict_fn` that returns class probabilities.
3. Define candidate shortcut interventions, such as flipping `source: alpha` to `source: beta`.
4. Compute confidence and CIC with `CICScorer`.
5. Assign the reliability quadrant with `ReliabilityPlane`.
6. Export an audit report under `results/audit_demo/`.

The demo writes:

- `results/audit_demo/cic_audit_certificates.csv`
- `results/audit_demo/cic_audit_summary.md`
- `results/audit_demo/cic_audit_report.json`
- `results/audit_demo/reliability_plane_audit.png`
- `results/audit_demo/reliability_plane_audit.pdf`

## Repair Extension

For settings where the candidate shortcut interventions are already available, the repair API can evaluate shortcut-neutralized counterfactuals and either return a repaired prediction, aggregate counterfactual consensus, stability-weight predictions, or abstain for human review:

```python
from causal_reliability.repair import repair_batch
```

The repair certificate keeps the original prediction, confidence, CIC score, quadrant, selected intervention, repaired prediction, action, and success flag. Abstentions are explicitly reported and should not be described as automatic corrections.

## CIC-Guided Abstention and Repair

### Motivation

CIC-guided repair is a proof-of-concept. The strongest current result is not universal automatic correction, but targeted detection and selective abstention: CIC can identify high-confidence shortcut failures that random augmentation fails to detect.

### Automatic Repair Results

Automatic correction means the repaired prediction becomes correct. It is reported separately from abstention. Real text repair supports controlled dangerous-quadrant correction, but not broad text-model repair.

### Selective Abstention Results

Selective abstention means the model does not make an automatic prediction. The abstention policy flags high-confidence, low-stability examples, low-confidence uncertainty, and cases with no valid counterfactual.

### What The Repair Extension Proves

When candidate interventions are available, CIC can guide human-review flags and selective-risk analysis for high-confidence shortcut failures.

### What It Does Not Prove

It does not prove CIC dramatically repairs all failures, dominates all baselines, or works on pretrained CLIP repair unless pretrained CLIP actually loaded and the repair run is marked `headline_eligible`.

## Non-Oracle CLIP Repair Boundary

The known-overlay CLIP repair experiment is an oracle upper bound: the true overlay bbox is supplied after benchmark generation and neutralized directly. That result can confirm that removing the known shortcut restores performance, but it is not evidence of automatic shortcut discovery.

The non-oracle CLIP repair experiment instead generates candidate regions from pixels, scores them without true labels or overlay bbox, and evaluates overlap only after ranking. This is still a finite-candidate audit workflow, not open-world discovery.

## Hard Multi-Decoy CLIP Headline

On a held-out hard multi-decoy benchmark using real pretrained OpenCLIP, misleading text reduced accuracy to 21.9%. Non-oracle CIC region scoring repaired accuracy to 87.5%, compared with 28.8% for matched random text-region repair, while preserving no-overlay accuracy and keeping clean-safe accuracy drop to 1.0%.

This does not solve open-world shortcut discovery. The method searches a finite candidate class of text-region proposals. Localization is strongest at coarse IoU >= 0.3 and weak at strict IoU >= 0.5, so the result should be interpreted as coarse causal-region localization and repair, not exact bounding-box recovery.

Oracle CLIP repair is upper-bound causal confirmation. Single-overlay non-oracle repair is promising but matched/random patch baselines can be competitive. The first multi-decoy repair is not a true shortcut-failure benchmark because original misleading accuracy was high. Hard multi-decoy repair is the main headline result.
