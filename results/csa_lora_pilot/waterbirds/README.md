# `results/csa_lora_pilot/waterbirds/`

Outputs of the **pre-registered Waterbirds CSA manual-LoRA pilot**
(`experiments/run_csa_lora_waterbirds.py`). See `docs/csa_lora_waterbirds.md` for
the full pre-registered design.

## Bounded scope / non-claims

This is **one bounded experiment**: does CSA, applied as a small manual-LoRA
adaptation of a real OpenCLIP visual tower, improve **worst-group robustness** on
real WILDS Waterbirds **without using group labels for CSA training**? Positive,
null, and negative results are all reported honestly.

It is **not** universal robustness, **not** open-world shortcut discovery,
**not** an RLHF/DPO replacement, **not** deployment validation, and **not** a
replacement for the finalized STS report.

## Files written here

| file | contents |
|---|---|
| `metrics.json` | full structured metrics, go/no-go flags, intervention summary, per-seed primary metrics |
| `table.csv` | per-mode average / worst-group / CIC instability / per-group accuracies |
| `group_table.csv` | long-form per-mode, per-group accuracy + counts |
| `summary.md` | human-readable bounded summary |
| `per_seed_metrics.csv` | per-seed primary-split metrics (only when >1 seed runs) |

## Honesty guarantees baked into the runner

- **Group labels are used only for evaluation** (per-group / worst-group
  accuracy) and the optional, clearly-marked Group DRO baseline — never by CSA
  training.
- CSA interventions are **finite diagnostic interventions** (label-free
  background/region perturbations), **not** verified Waterbirds causal masks.
- The `cached_embedding_adapter` fallback (used when no GPU/MPS is available) is
  **diagnostic only** and can never set `waterbirds_csa_promising` or
  `waterbirds_csa_strong` to true.
- Missing dataset / accelerator / OpenCLIP ⇒ a clean skip with
  `waterbirds_csa_null=true`.

## Not committed

Per `.gitignore`: `cache/`, `*.pt`, `*.pth`, `checkpoints/`, and all WILDS
downloaded data (`data/wilds/`). No checkpoints, raw datasets, or image caches
are committed.
