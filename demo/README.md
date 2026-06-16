# CIC finite-candidate reliability demo

A lightweight, presentation-ready demo of the **Causal Intervention Consistency
(CIC)** workflow, for STS judging and small external usability feedback.

> **Scope & limitations.** CIC tests *finite candidate interventions*. It is
> **not** guaranteed open-world shortcut discovery or universal robustness. This
> demo illustrates the *workflow*; it is **not** an experiment and does **not**
> produce scientific evidence. In mock mode the classifier is a deterministic
> stub — its numbers are illustrative only.

This demo does **not** modify the research report, change experimental metrics,
alter support gates, or write to `results/final_report/`.

---

## What it shows

The demo walks through the CIC reliability flow on a single image:

1. **Original prediction** — top-5 zero-shot labels + confidences.
2. **Finite candidate proposals** — model-free region proposals (the real
   `generate_open_region_proposals`).
3. **CIC scoring** — proposals are ranked by the real
   `score_region_candidates` (neutralize → measure prediction instability).
4. **Selected CIC region** — the highest-scoring candidate, outlined on the image.
5. **Repaired prediction** — top-5 after neutralizing the selected region
   (real `shortcut_neutralized_prediction`).
6. **Reliability / abstention** — accept the repair, or **abstain** when a
   high-confidence prediction is unstable (real `abstention_decision`).
7. **Export** — a small JSON + PNG report.

The proposal generation, region scoring, neutralization, and repair/abstention
logic are the **real project components**. Only the *classifier* is swappable.

---

## Modes

Configured in [`demo_config.yaml`](demo_config.yaml) (`mode:`), overridable on
the CLI.

| mode   | classifier | notes |
| ------ | ---------- | ----- |
| `mock` (default) | deterministic, model-free stub | safe, fast, no downloads. **Not scientific evidence.** |
| `real` | OpenCLIP zero-shot (real weights) | needs `open_clip_torch`; first run needs `--allow-download`. |
| `auto` | real if available, else mock | records which path ran. |

If real OpenCLIP dependencies are available, the real model path is used; if not,
the demo clearly labels itself as running in mock/demo mode.

---

## Launch locally

```bash
# 1) (optional) real model support
pip install gradio open_clip_torch

# 2a) mock mode (default — no downloads, deterministic)
python3 demo/app.py

# 2b) real OpenCLIP zero-shot (first run downloads ViT-B-32 weights)
python3 demo/app.py --mode real --allow-download
```

Then open the printed local URL. Upload an image or pick a **sample image**,
click **Run CIC**, and optionally **Export report**.

`gradio` is only needed to launch the UI. The pipeline
([`cic_pipeline.py`](cic_pipeline.py)) imports and runs without it, e.g.:

```python
from PIL import Image
from demo.cic_pipeline import DemoConfig, run_pipeline, export_report

result = run_pipeline(Image.open("demo/sample_images/text_overlay_success.png"),
                      DemoConfig(mode="mock"))
print(result.original_top_k[0], "->", result.repaired_top_k[0], result.reliability_action)
export_report(Image.open("demo/sample_images/text_overlay_success.png"),
              result, "results/demo_validation/exports")
```

---

## Sample images

Small, **synthetic, non-sensitive** images (generated, not copied from any
dataset). No raw COCO, `train2014`, or zip files ship in this demo. Regenerate
with `python3 demo/sample_images/generate_samples.py`. See
[`sample_images/manifest.json`](sample_images/manifest.json).

| file | scenario | mock-mode behavior |
| ---- | -------- | ------------------ |
| `text_overlay_success.png` | controlled text-overlay | shortcut → object **repair** |
| `semantic_decoy_success.png` | semantic decoy | shortcut → object **repair** |
| `coco_text_directional.png` | COCO-Text directional scene-text | finite-candidate case with **no confident repair** (illustrates the scope limit) |
| `failure_abstain.png` | failure / abstain | high-confidence but unstable → **abstain** |

The mix (two repairs, one no-op, one abstain) is intentional: a credible
reliability tool does not flip every prediction.

---

## Files

- [`cic_pipeline.py`](cic_pipeline.py) — importable pipeline + export (no gradio).
- [`app.py`](app.py) — Gradio UI (lazy-imports gradio; safe to import in tests).
- [`demo_config.yaml`](demo_config.yaml) — demo configuration.
- [`sample_images/`](sample_images/) — synthetic sample inputs + generator.

## External validation scaffold

For collecting small, bounded **external demonstration feedback**, see
[`../results/demo_validation/`](../results/demo_validation/):

- `external_validation_protocol.md` — how to run a small usability check.
- `external_validation_form.md` — the reviewer question form.
- `external_validation_template.csv` — a row-per-response template.

This is a *small usability check* / *external demonstration feedback* — **not**
deployment validation, clinical validation, or proof of real-world robustness.
