# Waterbirds CSA Manual-LoRA Pilot (pre-registered)

## Bounded question

Does **Counterfactual Stability Alignment (CSA)**, applied as a small
**manual-LoRA** adaptation of a real OpenCLIP visual tower, improve
**worst-group robustness** on the real WILDS Waterbirds dataset **without using
group labels for CSA training**?

This is **one bounded experiment**, pre-registered below, not a benchmark search
or a hyperparameter hunt. A positive, null, or negative result is acceptable and
is reported honestly. We do **not** run additional Waterbirds variants after
seeing results.

## Explicit non-claims

- This is **not** universal robustness.
- This is **not** open-world shortcut discovery.
- This is **not** an RLHF/DPO replacement.
- This is **not** deployment validation.
- This is **not** a replacement for the finalized STS report.

## Background and honesty constraints

A previous manual-LoRA CSA pilot on controlled text-overlay shortcuts worked, but
a pre-registered semantic-decoy transfer test was **null** against counterfactual
augmentation. Therefore this Waterbirds experiment is deliberately honest,
pre-specified, and bounded:

- **Group labels (`y` × `place`) are used only for evaluation** (per-group and
  worst-group accuracy) and for an *optional, off-by-default, clearly-marked*
  Group DRO baseline. They are **never** used by the CSA training objective.
- The CSA interventions are **finite diagnostic interventions** — label-free
  background/region perturbations — and are **not** verified ground-truth
  Waterbirds causal masks.
- Real `manual_lora_visual` requires a GPU/MPS accelerator and a loadable
  OpenCLIP backbone. With neither, the pilot **skips full LoRA cleanly** and runs
  a labelled `cached_embedding_adapter` fallback that is **diagnostic only** and
  can never set `waterbirds_csa_promising` or `waterbirds_csa_strong` to true.

## Dataset

- Uses **local WILDS Waterbirds artifacts** if present (`data/wilds/waterbirds_v*`
  with a `metadata.csv`). No download happens by default; pass `--download` to
  permit a one-time WILDS download.
- If the dataset is missing the pilot **skips cleanly** and writes
  `waterbirds_available=false` with `waterbirds_csa_null=true`.
- Object label `y`: 0 = landbird, 1 = waterbird. Background `place`: 0 = land,
  1 = water. The four evaluation groups are
  `landbird_on_land`, `landbird_on_water`, `waterbird_on_land`,
  `waterbird_on_water`. Worst-group accuracy is the minimum per-group accuracy
  over the groups present in the split.

## Model / adaptation

- Backbone: real OpenCLIP **ViT-B-32 / laion2b_s34b_b79k** from the local cache
  if available (no download by default).
- Mode: **`manual_lora_visual`** — actual LoRA on the visual tower via the
  internal `LoRALinear` wrapper (no PEFT). The text/prompt tower is **frozen**.
- Patch the **last 2 visual transformer blocks**, target modules `c_fc`,
  `c_proj`, `out_proj`, **rank 4**, **alpha 8**, **lr 2e-4**. No full-backbone
  fine-tuning. Only LoRA factors are trainable.
- Classifier: frozen zero-shot text head over the two prompts
  `"a photo of a landbird"` / `"a photo of a waterbird"` (zero-shot degradation
  stays measurable).

### Fallback (`cached_embedding_adapter`)

When no GPU/MPS is available (and `--force-cpu-lora` is not set), the pilot
encodes **frozen** OpenCLIP image embeddings (cached to
`results/csa_lora_pilot/waterbirds/cache/`) and trains a lightweight residual
adapter/head over them. This is **clearly labelled `cached_embedding_adapter`,
not LoRA**, is diagnostic only, and forces `waterbirds_csa_promising=false`,
`waterbirds_csa_strong=false`, `waterbirds_csa_null=true`.

## CSA interventions for Waterbirds (finite diagnostic candidates)

Waterbirds has no obvious text-overlay intervention, so we define a small finite
candidate bank of **label-free** perturbations and document them exactly:

| type | description |
|---|---|
| `region_mask` | fill a random rectangular region (side fraction `mask_frac`) with neutral gray |
| `region_blur` | average-pool blur applied inside a random rectangular region |
| `lowfreq_bg`  | add a smooth low-frequency field (coarse grid upsampled) over the whole image |
| `weak_crop`   | bounded object-preserving weak crop (`crop_min`..1.0) resized back |

These are **finite diagnostic interventions, not verified Waterbirds causal
masks**, and they do not use group labels. The exact candidate set used is
recorded in `metrics.json` (`intervention_summary`) and the summary.

## CSA objective

\[
L = L_{task} + \lambda_{stab} L_{stability} + \lambda_{cic} L_{CIC} + \lambda_{preserve} L_{preserve}
\]

- `L_task` — cross-entropy on the object label `y` (never the group).
- `L_stability` — `KL(clean || intervened)`: predictions should not move under the
  finite interventions. On Waterbirds the "clean" anchor is the original image
  (there is no overlay to remove).
- `L_CIC` — counterfactual-instability penalty (information radius) across the
  finite candidate bank — the CIC signal reused as a training loss.
- `L_preserve` — anchor the prediction to the frozen zero-shot reader captured
  before training, so clean/zero-shot accuracy does not collapse.

Defaults: `lambda_stability=1.0`, `lambda_cic=1.0`, `lambda_preserve=0.5`,
`preservation_mode=kl`.

## Baselines

1. **frozen** — frozen OpenCLIP zero-shot / prompt classifier.
2. **plain_ft** — plain manual-LoRA fine-tuning with task loss only.
3. **csa** — CSA manual-LoRA (the method under test).
4. **cf_aug** *(optional, `--enable-cf-aug`)* — counterfactual-augmentation
   manual-LoRA over the same finite candidate bank.
5. **group_dro** *(optional, `--group-dro`)* — Group DRO baseline. **This baseline
   is group-label-supervised** and is clearly marked as such; CSA does not use
   group labels, so the comparison is reported for context, not as like-for-like.

## Evaluation

Train on the Waterbirds **train** split; evaluate on **val/test** if present
(`primary_eval_split` selects the split used for go/no-go, default `test`).
Reported per mode:

- average accuracy;
- worst-group accuracy;
- per-group accuracies (landbird/land, landbird/water, waterbird/land,
  waterbird/water);
- clean / zero-shot degradation (vs frozen zero-shot);
- CIC instability before (plain) / after (CSA) under the finite interventions;
- trainable parameter count, patched module names, runtime, and device.

## Seeds

- Default: **seed 0 only** (first Colab/T4 pilot), marked `single_seed_pilot=true`.
- `--seeds 0,1,2` runs the full multi-seed pilot. Strong success cannot be
  claimed from a single seed.

## Pre-registered go / no-go

`waterbirds_csa_promising = true` **only if all**:

1. real `manual_lora_visual` was used (not the cached fallback);
2. average-accuracy drop vs plain manual-LoRA ≤ **0.03**;
3. worst-group accuracy improves by at least **+0.05** over plain manual-LoRA;
4. CSA reduces measured CIC instability by at least **20%** vs plain manual-LoRA.

`waterbirds_csa_strong = true` **only if**:

- seeds 0, 1, 2 complete (real LoRA);
- the mean paired (CSA − plain) worst-group gain ≥ **+0.08**;
- that mean paired gain **exceeds** the seed-to-seed standard deviation of the
  paired gains;
- the mean average-accuracy drop ≤ 0.03;
- if Group DRO was run, CSA is compared to it honestly while noting CSA did not
  use group labels.

`waterbirds_csa_null = true` if any of:

- worst-group gain < +0.05;
- average accuracy drops by > 0.03;
- instability improves but group robustness does not;
- only the cached fallback ran;
- the dataset or GPU/MPS (or OpenCLIP) was unavailable.

## Outputs

All under `results/csa_lora_pilot/waterbirds/`:

- `metrics.json` — full structured metrics, go/no-go, intervention summary,
  per-seed primary metrics.
- `table.csv` — per-mode average / worst-group / CIC instability / per-group.
- `group_table.csv` — long-form per-mode, per-group accuracy and counts.
- `summary.md` — the human-readable bounded summary.
- `per_seed_metrics.csv` — written when more than one seed runs.

No checkpoints, raw datasets, image caches, or WILDS downloads are committed
(see `.gitignore`).

## How to run

```bash
# Single-seed pilot (uses local data + cached OpenCLIP if present; auto device).
python3 experiments/run_csa_lora_waterbirds.py

# Full multi-seed pilot.
python3 experiments/run_csa_lora_waterbirds.py --seeds 0,1,2

# CPU smoke (explicitly opt into slow CPU manual-LoRA on a tiny subset).
python3 experiments/run_csa_lora_waterbirds.py --device cpu --force-cpu-lora \
    --max-train-examples 32 --max-eval-examples 32 --epochs 1
```

Tests (CPU-safe; never load real OpenCLIP or the real dataset):

```bash
python3 -m pytest tests/test_csa_lora_waterbirds.py
```
