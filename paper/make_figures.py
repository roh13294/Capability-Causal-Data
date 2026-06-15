#!/usr/bin/env python3
"""Generate publication-style figures for the paper from repository artifacts.

All numbers are taken from the repository's final artifacts (see comments).
No seaborn. Simple matplotlib only. Outputs PDF + PNG into paper/figures/.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

GREY = "#888888"
BLUE = "#2b6cb0"
GREEN = "#2f855a"
RED = "#c53030"
ORANGE = "#dd6b20"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 1: Reliability plane (conceptual two-axis diagram)
# ---------------------------------------------------------------------------
def fig_reliability_plane():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.axhline(0.5, color="k", lw=0.8)
    ax.axvline(0.5, color="k", lw=0.8)
    # quadrant shading
    ax.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, color=GREEN, alpha=0.12))   # high conf, high stab
    ax.add_patch(Rectangle((0.0, 0.5), 0.5, 0.5, color=BLUE, alpha=0.12))    # low conf, high stab
    ax.add_patch(Rectangle((0.0, 0.0), 0.5, 0.5, color=ORANGE, alpha=0.12))  # low conf, low stab
    ax.add_patch(Rectangle((0.5, 0.0), 0.5, 0.5, color=RED, alpha=0.16))     # high conf, low stab

    ax.text(0.75, 0.75, "RELIABLE\nhigh confidence\nhigh stability",
            ha="center", va="center", color=GREEN, fontweight="bold")
    ax.text(0.25, 0.75, "UNCERTAIN\nbut STABLE\nlow confidence\nhigh stability",
            ha="center", va="center", color=BLUE, fontweight="bold")
    ax.text(0.25, 0.25, "FRAGILE\nlow confidence\nlow stability",
            ha="center", va="center", color=ORANGE, fontweight="bold")
    ax.text(0.75, 0.25, "DANGEROUS\nSHORTCUT RELIANCE\nhigh confidence\nlow stability",
            ha="center", va="center", color=RED, fontweight="bold")

    ax.set_xlabel("Confidence  (how sure the model is)")
    ax.set_ylabel("Counterfactual stability  (1 - CIC)")
    ax.set_title("The Reliability Plane")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    save(fig, "reliability_plane")


# ---------------------------------------------------------------------------
# Figure 2: Hard multi-decoy main result (held-out benchmark)
# Source: results/final_report/final_key_numbers.json
# ---------------------------------------------------------------------------
def fig_main_results():
    labels = ["Misleading\n(original)", "Random\nmatched text", "CIC top-1",
              "CIC clean-safe", "Oracle\n(upper bound)", "No-overlay\n/ aligned"]
    vals = [0.25, 0.3306, 0.75, 0.75, 1.00, 1.00]
    colors = [RED, GREY, BLUE, BLUE, GREEN, GREEN]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Hard multi-decoy held-out benchmark (OpenCLIP ViT-B-32)")
    ax.tick_params(axis="x", labelsize=8)
    save(fig, "main_results")


# ---------------------------------------------------------------------------
# Figure 3: Failure-conditioned repair (n=50 verified failures)
# Source: results/hard_multidecoy_failure_conditioned/failure_conditioned_metrics.csv
# ---------------------------------------------------------------------------
def fig_failure_conditioned():
    labels = ["Original\n(by constr.)", "Random\nnontext patch", "Random\naugment.",
              "Random\nmatched text", "Largest\ntext", "Highest\ntextness",
              "CIC\nclean-safe", "CIC\ntop-1", "CIC\ntop-3", "Oracle"]
    vals = [0.00, 0.0024, 0.02, 0.112, 0.26, 0.82, 0.94, 0.96, 0.98, 1.00]
    colors = [RED, GREY, GREY, GREY, ORANGE, ORANGE, BLUE, BLUE, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, vals, color=colors, width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Repaired accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Failure-conditioned repair on 50 verified shortcut failures")
    ax.tick_params(axis="x", labelsize=7.5)
    save(fig, "failure_conditioned")


# ---------------------------------------------------------------------------
# Figure 4: Per-input class-balance (median residual to clean vs repair acc)
# Source: results/per_input_class_balance/per_input_class_balance_key_numbers.json
# ---------------------------------------------------------------------------
def fig_class_balance():
    conds = ["Random\nmatched", "CIC top-1", "CIC top-3", "Oracle"]
    residual = [5.220, 3.704, 3.506, 2.464]   # median residual to clean (lower = better)
    repair = [0.406, 0.781, 0.781, 1.000]      # repair accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))

    b1 = ax1.bar(conds, residual, color=[GREY, BLUE, BLUE, GREEN], width=0.6)
    ax1.axhline(2 * 3.0, color=RED, ls="--", lw=1.0, label=r"$2\epsilon_B$ (=6.0)")
    for b, v in zip(b1, residual):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Median class-balance residual (logits)")
    ax1.set_title("Lower residual = more class-balanced")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.legend(fontsize=8, frameon=False)

    b2 = ax2.bar(conds, repair, color=[GREY, BLUE, BLUE, GREEN], width=0.6)
    for b, v in zip(b2, repair):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Repair accuracy")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Repair accuracy")
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("Per-input class-balance tracks repair success (OpenCLIP text overlays)",
                 fontsize=10)
    save(fig, "class_balance")


# ---------------------------------------------------------------------------
# Figure 5: Object-entanglement schematic
# Source: results/embedding_additivity/embedding_additivity_key_numbers.json
# ---------------------------------------------------------------------------
def fig_object_entanglement():
    groups = ["Text overlay", "Watermark"]
    within_shortcut = [0.765, 0.757]
    within_object = [0.855, 0.923]
    shuffled = [0.634, 0.721]
    x = range(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar([i - w for i in x], shuffled, width=w, label="shuffled baseline", color=GREY)
    ax.bar([i for i in x], within_shortcut, width=w, label="within-shortcut cohesion", color=BLUE)
    ax.bar([i + w for i in x], within_object, width=w, label="within-object cohesion", color=ORANGE)
    for i in x:
        ax.text(i - w, shuffled[i] + 0.01, f"{shuffled[i]:.2f}", ha="center", fontsize=7.5)
        ax.text(i, within_shortcut[i] + 0.01, f"{within_shortcut[i]:.2f}", ha="center", fontsize=7.5)
        ax.text(i + w, within_object[i] + 0.01, f"{within_object[i]:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean pairwise cosine of embedding deltas")
    ax.set_ylim(0, 1.05)
    ax.set_title("Object-entanglement: shortcut shift clusters by object, not shortcut value")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, "object_entanglement")


if __name__ == "__main__":
    fig_reliability_plane()
    fig_main_results()
    fig_failure_conditioned()
    fig_class_balance()
    fig_object_entanglement()
    print("All figures written to", FIGDIR)
