# Human label-preservation validation

This directory holds the human study that tests whether CIC neutralization is
label-preserving for human viewers: annotators compare original vs. repaired
(neutralized) image pairs and judge whether the object label is preserved and
whether the repaired image is still recognizable.

## Layout

```
validation/human_label_preservation/
  analyze_annotations.py            # analysis + Fleiss' kappa (no extra deps)
  completed_annotations/            # one CSV per annotator (raw responses)
  packet/metadata_hidden.csv        # per-pair true labels + image paths (hidden from annotators)
```

## Running

```bash
python3 validation/human_label_preservation/analyze_annotations.py
```

Outputs are written to `results/human_label_preservation/`:

* `human_validation_summary.md`
* `human_validation_metrics.json`
* `human_validation_flags.csv`

## Expected raw-annotation columns

Each file in `completed_annotations/` is one annotator's responses, with columns
(aliases tolerated):

| column | meaning |
| --- | --- |
| `annotator_id` | annotator identifier (falls back to filename) |
| `example_id` | pair id, joins to `packet/metadata_hidden.csv` |
| `before_label` | object label seen in the original image |
| `after_label` | object label seen in the repaired image |
| `label_changed` | did the object label change? yes/no |
| `after_recognizable` | is the repaired image recognizable? yes/no |
| `unsure` | optional: annotator was unsure |
| `notes` | optional free text |

When these files are present the script computes majority-vote before/after label
accuracy against the true label, label-preservation and recognizability rates, the
unsure rate, percent agreement, and Fleiss' kappa for four fields (before label,
after label, label-change, recognizability), and it enumerates and characterizes
the specific pairs where majority vote did not preserve the label.

## Current data status (important)

The raw per-annotator response files were **not retained** in this checkout. Only
the study aggregates were recorded:

* 3 annotators, 100 original/repaired pairs
* majority-vote object label preserved in **96/100** pairs
* repaired image recognizable in **97/100** pairs
* 4 preservation failures (retained and flagged, not removed)

Because the raw responses are unavailable, the script reports these aggregates but
marks Fleiss' kappa, percent agreement, before/after label accuracy, the unsure
rate, and the identity of the four failing pairs as unavailable rather than
estimating them. **No per-annotator annotations are fabricated.** Dropping the raw
responses into `completed_annotations/` (and `packet/metadata_hidden.csv`) and
re-running the script recomputes all metrics from the raw data.
