"""Gradio front-end for the CIC finite-candidate reliability demo.

Run locally:
    pip install gradio open_clip_torch   # gradio is required only to launch
    python3 demo/app.py                  # mock mode by default
    python3 demo/app.py --mode real --allow-download   # real OpenCLIP weights

This module imports cleanly WITHOUT gradio installed and WITHOUT launching a
server; gradio is imported lazily inside ``launch()``/``build_interface()`` so
tests can import the demo and exercise the pipeline headlessly.

Nothing here is an experiment. It does not read or write benchmark result
folders, support gates, or results/final_report/.

Scope note (always shown in the UI):
    CIC tests finite candidate interventions. It is not guaranteed open-world
    shortcut discovery or universal robustness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from demo.cic_pipeline import (
    DEMO_DISCLAIMER,
    SCOPE_NOTE,
    DemoConfig,
    PipelineResult,
    export_report,
    render_region_overlay,
    run_pipeline,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "demo_config.yaml"


def load_config(path: str | Path | None = None) -> DemoConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path.exists():
        return DemoConfig.from_yaml(path)
    return DemoConfig()


def list_sample_images(config: DemoConfig) -> list[str]:
    sample_dir = Path(config.sample_images_dir)
    if not sample_dir.is_absolute():
        sample_dir = HERE.parent / sample_dir
    if not sample_dir.exists():
        return []
    return sorted(str(p) for p in sample_dir.glob("*.png"))


def _topk_markdown(title: str, rows: list[dict[str, float]]) -> str:
    lines = [f"**{title}**", "", "| rank | label | confidence |", "| --- | --- | --- |"]
    for i, row in enumerate(rows, start=1):
        lines.append(f"| {i} | {row['label']} | {row['confidence']:.3f} |")
    return "\n".join(lines)


def format_summary(result: PipelineResult) -> str:
    """Human-readable markdown summary for the UI."""
    orig = result.original_top_k[0] if result.original_top_k else {"label": "n/a", "confidence": 0.0}
    rep = result.repaired_top_k[0] if result.repaired_top_k else {"label": "n/a", "confidence": 0.0}
    mode_banner = (
        "🟢 **REAL model path** (OpenCLIP)"
        if result.mode_used == "real"
        else "🟡 **MOCK/demo mode** — deterministic stub classifier. Outputs are NOT scientific evidence."
    )
    reliability = (
        "⚠️ **ABSTAIN** — high-confidence but unstable under intervention; the gate withholds a repaired label."
        if result.reliability_action == "abstain"
        else "✅ **Accept repair** — prediction is stable under the selected intervention."
    )
    parts = [
        mode_banner,
        "",
        f"**Backend:** `{result.backend}`  |  **candidates scored:** {result.n_candidates}",
        "",
        _topk_markdown("1. Original top-5", result.original_top_k),
        "",
        f"**2. CIC selected region:** `{result.selected_region.get('proposal_type', 'n/a')}` "
        f"bbox={result.selected_region.get('bbox', [])}  "
        f"(CIC score {result.cic_selected_score:.3f})",
        "",
        _topk_markdown("3. Repaired top-5 (selected counterfactual)", result.repaired_top_k),
        "",
        f"**Prediction changed:** {result.prediction_changed}  |  "
        f"**stability:** {result.stability_score:.2f}  |  "
        f"**strategy:** `{result.repair_strategy}`",
        "",
        f"**4. Reliability gate:** {reliability}",
        "",
        f"_Original top-1_: **{orig['label']}** ({orig['confidence']:.2f}) → "
        f"_repaired top-1_: **{rep['label']}** ({rep['confidence']:.2f})",
        "",
        "---",
        f"**Scope & limitations:** {SCOPE_NOTE}",
        "",
        f"_{DEMO_DISCLAIMER}_",
    ]
    return "\n".join(parts)


def analyze(image, config: DemoConfig):
    """Core callback: returns (overlay_image, summary_markdown, json_dict)."""
    if image is None:
        return None, "Upload or select an image to run the CIC demo.", {}
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    result = run_pipeline(image, config)
    overlay = render_region_overlay(image, result)
    return overlay, format_summary(result), result.to_dict()


def build_interface(config: DemoConfig | None = None):
    """Construct (but do not launch) the Gradio Blocks app. Imports gradio lazily."""
    import gradio as gr

    config = config or load_config()
    samples = list_sample_images(config)

    def _run(image):
        overlay, summary, payload = analyze(image, config)
        return overlay, summary, payload

    def _export(image):
        if image is None:
            return None
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        result = run_pipeline(image, config)
        paths = export_report(image, result, config.export_dir)
        return paths.get("png")

    with gr.Blocks(title="CIC finite-candidate reliability demo") as app:
        gr.Markdown(
            "# CIC finite-candidate reliability demo\n"
            f"> **Scope & limitations:** {SCOPE_NOTE}\n\n"
            f"_{DEMO_DISCLAIMER}_"
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload or select an image")
                if samples:
                    gr.Examples(examples=[[s] for s in samples], inputs=inp, label="Sample images")
                run_btn = gr.Button("Run CIC", variant="primary")
                export_btn = gr.Button("Export report (JSON + PNG)")
            with gr.Column():
                overlay_out = gr.Image(label="CIC proposal visualization (selected region)")
                summary_out = gr.Markdown()
                json_out = gr.JSON(label="Structured result")
                export_out = gr.File(label="Exported report")

        run_btn.click(_run, inputs=inp, outputs=[overlay_out, summary_out, json_out])
        export_btn.click(_export, inputs=inp, outputs=export_out)
    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the CIC demo (Gradio).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--mode", choices=["mock", "real", "auto"], default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args(argv)


def launch(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    if args.mode:
        config.mode = args.mode
    if args.allow_download:
        config.allow_download = True
    app = build_interface(config)
    app.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    launch()
