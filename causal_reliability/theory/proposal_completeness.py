from __future__ import annotations

"""Checkable encoding of the proposal-complete CIC repair theory.

This module is a *consistency check* of the theorem statements in
``docs/global_cic_theory_attempt.md``. It implements the core inequalities so the
unit tests can confirm that the logical guarantees hold exactly as proved (and,
crucially, that they *fail* when their assumptions are violated — including a
numeric witness of the no-free-lunch / impossibility construction).

It introduces **no** experimental metric and is not imported by any experiment
runner. It operates on logit vectors only.

Notation (matches the docs):

* ``clean_logits``      : ``ell_y(x_clean)``   — clean/causal logits.
* ``repaired_logits``   : ``ell_y(T_a(x))``    — logits after applying intervention a.
* ``rho_y = repaired - clean``                 — residual-to-clean.
* ``eps_B``             : per-input residual-to-clean class-imbalance budget.
"""

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray


# --------------------------------------------------------------------------- #
# Residual-to-clean class imbalance and margins
# --------------------------------------------------------------------------- #
def residual_to_clean(repaired_logits: ArrayLike, clean_logits: ArrayLike) -> np.ndarray:
    """``rho_y(x) = ell_y(T_a(x)) - ell_y(x_clean)`` (Section 2)."""

    return np.asarray(repaired_logits, dtype=float) - np.asarray(clean_logits, dtype=float)


def class_imbalance(residual: ArrayLike) -> float:
    """``max_y |rho_y - mean_y rho|`` — the per-input class-imbalance of a residual."""

    rho = np.asarray(residual, dtype=float)
    return float(np.max(np.abs(rho - rho.mean())))


def logit_margin(logits: ArrayLike) -> float:
    """Top-1 minus runner-up margin of a logit vector."""

    z = np.sort(np.asarray(logits, dtype=float))
    return float(z[-1] - z[-2])


def is_good_intervention(repaired_logits: ArrayLike, clean_logits: ArrayLike, eps_B: float) -> bool:
    """Section 2(4): an intervention is *good* iff its residual-to-clean is
    class-balanced within ``eps_B``."""

    return class_imbalance(residual_to_clean(repaired_logits, clean_logits)) <= eps_B + 1e-12


# --------------------------------------------------------------------------- #
# Lemma 2 / Lemma 3: recovery
# --------------------------------------------------------------------------- #
def recovery_certified_by_clean_margin(
    repaired_logits: ArrayLike, clean_logits: ArrayLike, eps_B: float
) -> bool:
    """Lemma 2: if the intervention is good and ``m_clean > 2 eps_B`` then the
    repaired argmax equals the clean argmax. Returns whether the *sufficient
    condition* holds (not merely whether the argmaxes happen to match)."""

    good = is_good_intervention(repaired_logits, clean_logits, eps_B)
    return bool(good and logit_margin(clean_logits) > 2.0 * eps_B)


def recovery_certified_by_repaired_margin(
    repaired_logits: ArrayLike, eps_B: float
) -> bool:
    """Lemma 3 (observable-margin form): if the intervention is good (class
    imbalance <= eps_B, supplied as the calibrated budget) and the *observed*
    repaired margin exceeds ``2 eps_B``, recovery is certified using only the
    repaired logits."""

    return bool(logit_margin(repaired_logits) > 2.0 * eps_B)


def argmaxes_agree(repaired_logits: ArrayLike, clean_logits: ArrayLike) -> bool:
    return int(np.argmax(repaired_logits)) == int(np.argmax(clean_logits))


# --------------------------------------------------------------------------- #
# Lemma 1: selection under a scoring margin and bounded noise
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    name: str
    noiseless_score: float
    is_good: bool


def top1_selects_good(candidates: list[Candidate], observed_scores: dict[str, float]) -> bool:
    """Whether the empirical top-1 (by ``observed_scores``) is a good intervention."""

    top = max(candidates, key=lambda c: observed_scores[c.name])
    return top.is_good


def selection_lemma_holds(candidates: list[Candidate], gamma: float, noise_bound: float) -> bool:
    """Lemma 1 premise check: at least one good candidate leads every harmful
    candidate by ``gamma`` noiselessly, and noise is below ``gamma/2``."""

    goods = [c for c in candidates if c.is_good]
    harms = [c for c in candidates if not c.is_good]
    if not goods:
        return False
    best_good = max(c.noiseless_score for c in goods)
    leads_all = all(best_good >= h.noiseless_score + gamma - 1e-12 for h in harms)
    return bool(leads_all and noise_bound <= gamma / 2.0 + 1e-12)


# --------------------------------------------------------------------------- #
# Theorem 2: observable success gate (calibrated)
# --------------------------------------------------------------------------- #
def gate_passes(
    *,
    score_gap_top1_median: float,
    repaired_margin: float,
    stability_gain: float,
    selected_area_fraction: float,
    clean_safe_score: float,
    thresholds: dict[str, float],
) -> bool:
    """Observable success gate ``G(x, a)`` (Section 5). Thresholds are *calibrated*
    on validation data; this function only evaluates the predicate."""

    return bool(
        score_gap_top1_median >= thresholds["min_score_gap"]
        and repaired_margin >= thresholds["min_repaired_margin"]
        and stability_gain >= thresholds["min_stability_gain"]
        and selected_area_fraction <= thresholds["max_area_fraction"]
        and clean_safe_score >= thresholds["min_clean_safe_score"]
    )


def gated_repair_certified(
    *, gate_ok: bool, repaired_logits: ArrayLike, eps_hat: float
) -> bool:
    """Theorem 2: gate passes (so eps_B <= eps_hat by calibration) AND the observed
    repaired margin exceeds ``2 eps_hat``."""

    return bool(gate_ok and recovery_certified_by_repaired_margin(repaired_logits, eps_hat))
