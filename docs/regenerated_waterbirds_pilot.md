# Regenerated Waterbirds-style spurious-background CIC pilot

This is **optional supporting evidence** for finite-candidate CIC repair on a
spurious-background failure mode. It is **not** a replacement for the main
OpenCLIP text-overlay headline result, and it is **not** open-world shortcut
discovery.

## What it does

The previous local Waterbirds pilot
([run_waterbirds_cic_pilot.py](../causal_reliability/experiments/run_waterbirds_cic_pilot.py))
needed a Waterbirds-style dataset with an oracle-repairable bird/background mask
but could not always find one. This pilot **regenerates** such a benchmark from
primary assets:

* **Birds:** CUB-200-2011 images plus their pixel-perfect segmentation masks.
* **Backgrounds:** Places land/water scenes.

For each bird we crop around its bounding box (with margin), then composite the
bird (using its exact binary mask) onto a Places background, holding the bird at
a fixed location so a single mask applies to all regimes:

* **aligned** — waterbird on water, landbird on land (shortcut helps).
* **misleading** — waterbird on land, landbird on water (shortcut hurts).
* **neutral** — bird on a neutral gray background.

Because the bird mask is known exactly, we get an **oracle background
intervention** (neutralize the background, keep the bird) as an upper bound, and
a transparent way to validate — *after scoring* — whether a candidate
intervention touched foreground or background.

## Class and background mapping

* **Bird label** (`landbird` / `waterbird`): uses an official mapping if one is
  found locally (optional `waterbird_classes.txt`); otherwise a transparent
  keyword heuristic over CUB class names (gull, tern, auklet, pelican,
  cormorant, frigatebird, loon, grebe, duck, goose, swan, …). When heuristic,
  artifacts record `class_map_kind: heuristic_keyword`.
* **Background label** (`water` / `land`): inferred from Places folder/scene
  names (lake, ocean, river, beach, …  vs. forest, field, mountain, …). If
  background labels cannot be inferred, the pilot skips with a clear message.

## Non-oracle CIC scoring (no leakage)

The non-oracle CIC scorer ([`cic_rank`](../causal_reliability/experiments/run_regenerated_waterbirds_cic.py))
receives **only** image pixels, CLIP probabilities/logits, and the class
prompts. It never receives the true bird label, the background label,
correctness, or any mask (bird/background/oracle). Masks are used only for:

1. dataset generation,
2. oracle background neutralization,
3. evaluation/metadata,
4. *post hoc* validation of whether a candidate hit foreground/background.

## Evaluation modes

* **Mode A (natural):** aligned / misleading / neutral / oracle / CIC top-1 /
  CIC top-k / matched-random accuracies, plus clean (aligned) preservation.
* **Mode B (failure-conditioned):** keep only examples where the aligned image
  is correct, the misleading image is wrong (confidence ≥ threshold, default
  0.50), and oracle background neutralization restores the correct label.
  *Original accuracy is 0 by construction on this subset.*

## Headline eligibility gate

`regenerated_waterbirds_headline_eligible` becomes `true` only if **all** hold:
real pretrained OpenCLIP loaded (not fake); actual CUB images + CUB masks +
Places backgrounds used; the non-oracle scorer excludes label/background/mask/
correctness; ≥100 natural OR ≥30 failure-conditioned examples; misleading
accuracy meaningfully below aligned; oracle improves accuracy by ≥0.15 absolute
OR restores ≥0.80 on failure-conditioned; CIC beats matched random by ≥0.15
absolute OR CIs do not overlap; clean/aligned preservation acceptable; and the
report uses "controlled regenerated Waterbirds-style benchmark", not "open-world
discovery". If any fails, the run is recorded as a pilot/negative and is **not**
added as a main positive result.

## Data placement

```
data/cub/CUB_200_2011/
  images.txt
  image_class_labels.txt
  classes.txt
  bounding_boxes.txt
  images/<class>/<file>.jpg
data/cub/segmentations/<class>/<file>.png   # CUB segmentation masks
data/places/<scene_name>/<file>.jpg         # folders named by scene (lake, forest, …)
```

`data.allow_download` defaults to `false`; the script never downloads large
datasets automatically (with `allow_download: true` it only **prints** the
placement instructions above). You may `pip install wilds` if you prefer, but
the preferred path is controlled regeneration from CUB + Places for pixel-perfect
masks.

## Run

```bash
python3 -m pytest tests/test_regenerated_waterbirds_cic.py
python3 -m causal_reliability.experiments.run_regenerated_waterbirds_cic \
  --config configs/regenerated_waterbirds_cic.yaml
```

If assets are missing the run skips cleanly and writes
`results/regenerated_waterbirds_cic/waterbirds_regeneration_summary.md` and
`waterbirds_regeneration_key_numbers.json` listing the missing paths.

## WILDS Waterbirds metadata-only diagnostic (motivation, not CIC repair)

Separately from the regenerated pilot above, WILDS Waterbirds itself was parsed as
a real spurious-background diagnostic (11,788 examples detected and parsed). A
metadata-only OpenCLIP evaluation showed the expected background sensitivity:

| group | accuracy |
| --- | --- |
| overall | 56.0% |
| land background | 73.1% |
| water background | 35.9% |
| landbird on land | 74.4% |
| landbird on water | 21.6% |

(Source: `results/waterbirds_cic_pilot/wilds_metadata_diagnostic.csv`.)

However, WILDS Waterbirds does **not** ship oracle-repairable bird/background masks
or bounding boxes, so CIC repair and failure-conditioned oracle repair were **not**
run on it. This diagnostic motivates a future regenerated CUB+Places Waterbirds-style
benchmark with known masks (the pilot described above), but it is **not** a positive
CIC repair result and must never be listed as headline evidence.

## Scope and integrity

This is a controlled regenerated benchmark, finite-candidate intervention only.
It does **not** claim real WILDS Waterbirds evaluation (unless those exact assets
were supplied), open-world discovery, exact localization, or general robustness,
and it does not change any existing OpenCLIP text-overlay result.
