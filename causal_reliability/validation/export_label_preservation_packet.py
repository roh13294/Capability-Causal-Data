from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def _text_pair(row: pd.Series) -> dict[str, Any]:
    return {
        "domain": "real_text_shortcut",
        "example_id": row.get("example_id", ""),
        "original_path_or_text": row.get("marked_text", ""),
        "counterfactual_path_or_text": row.get("counterfactual_text", ""),
        "original_true_label": row.get("label", ""),
        "intended_counterfactual_true_label": row.get("label", ""),
        "intervention": "flip neutral source marker",
        "intervention_type": "metadata_marker_flip",
        "expected_label_preserved": True,
        "label_preservation_rationale": "The review content is unchanged; only neutral source metadata changes.",
    }


def _default_pairs(results_dir: Path, max_per_domain: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(20, max(10, max_per_domain))
    digit_labels = list(range(10))
    digit_colors = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "pink", "gray", "lime"]
    for i in range(n):
        label = digit_labels[i % len(digit_labels)]
        old_color = digit_colors[i % len(digit_colors)]
        new_color = digit_colors[(i + 3) % len(digit_colors)]
        rows.append(
            {
                "domain": "colored_digits",
                "example_id": f"colored_digits_{i:03d}",
                "original_path_or_text": f"examples/colored_digits_{i:03d}_original.txt",
                "counterfactual_path_or_text": f"examples/colored_digits_{i:03d}_counterfactual.txt",
                "original_true_label": str(label),
                "intended_counterfactual_true_label": str(label),
                "intervention": f"recolor digit from {old_color} to {new_color}",
                "intervention_type": "digit_color_recolor",
                "expected_label_preserved": True,
                "label_preservation_rationale": "The digit identity is unchanged; only the shortcut color changes.",
                "original_render_text": f"Digit {label} rendered in {old_color}.",
                "counterfactual_render_text": f"The same digit {label} rendered in {new_color}.",
            }
        )
    shapes = ["circle", "square", "triangle", "star"]
    overlays = ["cat", "truck", "blue", "source A", "class zero"]
    for i in range(n):
        shape = shapes[i % len(shapes)]
        overlay = overlays[i % len(overlays)]
        cf_overlay = overlays[(i + 2) % len(overlays)]
        rows.append(
            {
                "domain": "clip_overlay",
                "example_id": f"clip_overlay_{i:03d}",
                "original_path_or_text": f"examples/clip_overlay_{i:03d}_original.txt",
                "counterfactual_path_or_text": f"examples/clip_overlay_{i:03d}_counterfactual.txt",
                "original_true_label": shape,
                "intended_counterfactual_true_label": shape,
                "intervention": f"change overlay text from {overlay} to {cf_overlay}",
                "intervention_type": "text_overlay_change",
                "expected_label_preserved": True,
                "label_preservation_rationale": "The underlying shape stays fixed; only the overlaid shortcut text changes.",
                "original_render_text": f"Image of a {shape} with overlaid text '{overlay}'.",
                "counterfactual_render_text": f"Same image of a {shape} with overlaid text '{cf_overlay}'.",
            }
        )
    text_path = results_dir / "real_text_shortcut" / "real_text_certificates.csv"
    if text_path.exists():
        text = pd.read_csv(text_path).head(n)
        rows.extend(_text_pair(row) for _, row in text.iterrows())
    else:
        texts = [
            (1, "The acting is warm and the story stays engaging."),
            (0, "The plot is thin and the pacing feels dull."),
            (1, "A witty script keeps the movie lively."),
            (0, "The characters are flat and the ending disappoints."),
            (1, "The direction is confident and the cast is charming."),
            (0, "The jokes miss and the scenes drag."),
            (1, "The soundtrack and performances are excellent."),
            (0, "The movie feels careless and forgettable."),
            (1, "A thoughtful drama with a satisfying finish."),
            (0, "The story is confusing and rarely convincing."),
            (1, "The film is energetic, funny, and sincere."),
            (0, "The thriller is predictable and badly edited."),
        ]
        markers = ["[SOURCE=A]", "[BATCH=17]", "[SITE=blue]"]
        for i in range(n):
            label, text = texts[i % len(texts)]
            marker = markers[i % len(markers)]
            cf_marker = markers[(i + 1) % len(markers)]
            rows.append(
                {
                    "domain": "real_text_shortcut",
                    "example_id": f"real_text_shortcut_{i:03d}",
                    "original_path_or_text": f"{marker} {text}",
                    "counterfactual_path_or_text": f"{cf_marker} {text}",
                    "original_true_label": "positive sentiment" if label else "negative sentiment",
                    "intended_counterfactual_true_label": "positive sentiment" if label else "negative sentiment",
                    "intervention": f"replace neutral metadata marker {marker} with {cf_marker}",
                    "intervention_type": "metadata_marker_replacement",
                    "expected_label_preserved": True,
                    "label_preservation_rationale": "The review content is unchanged; only neutral metadata changes.",
                }
            )
    return rows


def run(cfg: dict[str, Any]) -> dict[str, str]:
    results_dir = Path(cfg.get("results_dir", "results"))
    out_dir = ensure_dir(results_dir / "label_preservation_packet")
    examples_dir = ensure_dir(out_dir / "examples")
    max_per_domain = int(cfg.get("examples_per_domain", cfg.get("max_per_domain", 12)))
    pairs = pd.DataFrame(_default_pairs(results_dir, max_per_domain))
    pairs.insert(0, "pair_id", [f"lp_{i:04d}" for i in range(len(pairs))])
    pairs.to_csv(out_dir / "label_preservation_pairs.csv", index=False)
    for _, row in pairs.iterrows():
        original = row.get("original_render_text", row["original_path_or_text"])
        counterfactual = row.get("counterfactual_render_text", row["counterfactual_path_or_text"])
        (examples_dir / f"{row['pair_id']}.txt").write_text(
            "\n".join(
                [
                    f"Example ID: {row['example_id']}",
                    f"Domain: {row['domain']}",
                    f"Original: {original}",
                    f"Counterfactual: {counterfactual}",
                    f"Original reference label: {row['original_true_label']}",
                    f"Intended counterfactual reference label: {row['intended_counterfactual_true_label']}",
                    f"Rationale: {row['label_preservation_rationale']}",
                ]
            ),
            encoding="utf-8",
        )
    instructions = [
        "# Label-Preservation Validation Instructions",
        "",
        "For each pair, inspect the original example and the counterfactual example.",
        "",
        "Answer:",
        "1. What is the true label of the original example?",
        "2. What is the true label of the counterfactual example?",
        "3. Did the label stay the same?",
        "4. Is the counterfactual plausible?",
        "5. Any concerns?",
        "",
        "Use the provided label only as benchmark context; do not mark preservation true unless the counterfactual itself preserves the label.",
    ]
    (out_dir / "label_preservation_instructions.md").write_text("\n".join(instructions), encoding="utf-8")
    form = [
        "# Label-Preservation Response Template",
        "",
        "Required CSV columns:",
        "",
        "`annotator_id,example_id,original_label_human,counterfactual_label_human,label_preserved_human,plausible_human,concerns`",
        "",
        "Use `yes`/`no` for `label_preserved_human` and `plausible_human`.",
    ]
    (out_dir / "label_preservation_form_template.md").write_text("\n".join(form), encoding="utf-8")
    google = [
        "# Google Form Questions",
        "",
        "Create one form section per example. Include the original and counterfactual text/path exactly as listed in `label_preservation_pairs.csv`.",
        "",
        "1. What is the true label of the original?",
        "2. What is the true label of the counterfactual?",
        "3. Did the label stay the same?",
        "4. Is the counterfactual plausible?",
        "5. Any concerns?",
        "",
        "Export responses with columns: `annotator_id,example_id,original_label_human,counterfactual_label_human,label_preserved_human,plausible_human,concerns`.",
    ]
    (out_dir / "label_preservation_google_form_questions.md").write_text("\n".join(google), encoding="utf-8")
    return {
        "pairs": str(out_dir / "label_preservation_pairs.csv"),
        "instructions": str(out_dir / "label_preservation_instructions.md"),
        "form": str(out_dir / "label_preservation_form_template.md"),
        "google_form_questions": str(out_dir / "label_preservation_google_form_questions.md"),
        "examples": str(examples_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/label_preservation_packet.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
