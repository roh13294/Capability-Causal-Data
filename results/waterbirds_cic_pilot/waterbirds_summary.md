# Waterbirds finite-candidate CIC pilot (skipped)

This is an optional supporting pilot, not the main result.

**WILDS Waterbirds was found and parsed, but no oracle-repairable masks/bboxes were available, so the repair pilot is not headline-eligible.**

- Source: `wilds`
- Dataset available: `True` (11788 examples parsed)
- Oracle-repairable masks/bboxes available: `False`
- Oracle repair available: `False`
- Mask/seg/bbox-named files found in dataset tree: `0`
- CLIP backend: `not_checked` (pretrained loaded: `False`)
- Waterbirds headline eligible: `False`

- Converted WILDS metadata: `results/waterbirds_cic_pilot/wilds_converted_metadata.csv`
- Metadata-only diagnostic (NOT CIC repair): `results/waterbirds_cic_pilot/wilds_metadata_diagnostic.csv`

The pilot intentionally skips oracle and failure-conditioned repair rather
than fabricating a result. WILDS Waterbirds ships no bird/background masks or
bounding boxes, so oracle background neutralization is not possible. Provide a
dataset with segmentation masks or bird bounding boxes, then re-run. The main
OpenCLIP text-overlay headline result is unaffected.

This experiment only ever searches a finite, explicit candidate-intervention
set. It does not perform open-world discovery and does not claim general robustness.