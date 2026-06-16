from __future__ import annotations

"""Consistency checks for the *complete finite-candidate characterization of CIC
repair* (docs/theory.md Section 11; summarized in paper/main.tex).

The four pieces are checked as toy numeric witnesses:

1. exact repair criterion (and its residual-form equivalence),
2. tight residual-margin certificate (sufficiency AND boundary tightness),
3. proposal coverage ceiling (no scorer beats E[R_tau]),
4. a repair-localization conflict example (repair-optimal != localization-optimal).
"""

import numpy as np

from causal_reliability.theory.finite_candidate_characterization import (
    causal_margin,
    class_imbalance,
    coverage_ceiling_holds,
    coverage_indicator,
    expected_coverage,
    localization_success,
    margin_certificate_sufficient,
    repair_localization_conflict_example,
    repairs,
    repairs_via_residual,
    tightness_witness,
)


# --------------------------------------------------------------------------- #
# Piece 1: exact repair criterion
# --------------------------------------------------------------------------- #
def test_exact_repair_criterion_basic():
    # y* = 0 wins -> repairs; y* = 0 loses to class 1 -> does not.
    assert repairs(np.array([2.0, 1.0, 0.0]), y_star=0)
    assert not repairs(np.array([0.5, 2.0, 0.0]), y_star=0)
    # ties do not count as repair (strict separation required).
    assert not repairs(np.array([1.0, 1.0, 0.0]), y_star=0)


def test_repair_criterion_residual_form_matches_logit_form():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        clean = rng.normal(size=4)
        residual = rng.normal(size=4)
        y_star = int(rng.integers(0, 4))
        repaired = clean + residual
        assert repairs(repaired, y_star) == repairs_via_residual(clean, residual, y_star)


# --------------------------------------------------------------------------- #
# Piece 2: tight residual-margin certificate
# --------------------------------------------------------------------------- #
def test_margin_certificate_sufficiency():
    """If imbalance <= eps and m_clean > 2 eps, EVERY allowed residual repairs."""
    rng = np.random.default_rng(1)
    eps = 0.5
    checked = 0
    for _ in range(500):
        clean = rng.normal(size=5)
        y_star = int(np.argmax(clean))
        if not margin_certificate_sufficient(clean, y_star, eps):
            continue
        checked += 1
        # an arbitrary residual with class imbalance <= eps
        raw = rng.uniform(-1.0, 1.0, size=5)
        raw = raw - raw.mean()
        scale = class_imbalance(raw)
        residual = raw * (eps / scale) if scale > 0 else raw
        assert class_imbalance(residual) <= eps + 1e-9
        assert repairs(clean + residual, y_star)
    assert checked > 0  # the sufficient condition is actually exercised


def test_margin_certificate_tightness_at_boundary():
    """If m_clean <= 2 eps, the witness residual (imbalance <= eps) flips the top."""
    eps = 1.0
    # margin exactly 1.5 < 2*eps = 2.0
    clean = np.array([1.5, 0.0, 0.0])
    y_star = 0
    assert abs(causal_margin(clean, y_star) - 1.5) < 1e-9
    assert not margin_certificate_sufficient(clean, y_star, eps)
    r = tightness_witness(clean, y_star, eps)
    assert class_imbalance(r) <= eps + 1e-9          # the perturbation is allowed
    assert not repairs(clean + r, y_star)            # and it flips the top class
    assert int(np.argmax(clean + r)) != y_star


def test_margin_certificate_witness_imbalance_is_exactly_eps():
    eps = 0.7
    clean = np.array([0.4, 0.0, 0.0, 0.0])  # margin 0.4 < 2*eps
    r = tightness_witness(clean, y_star=0, eps=eps)
    assert abs(class_imbalance(r) - eps) < 1e-9


# --------------------------------------------------------------------------- #
# Piece 3: proposal coverage ceiling
# --------------------------------------------------------------------------- #
def test_coverage_indicator_and_localization_success():
    overlaps = np.array([0.1, 0.35, 0.6])
    assert coverage_indicator(overlaps, tau=0.5) == 1
    assert coverage_indicator(overlaps, tau=0.7) == 0
    assert localization_success(overlaps, chosen_idx=2, tau=0.5) == 1
    assert localization_success(overlaps, chosen_idx=0, tau=0.5) == 0


def test_no_scorer_beats_expected_coverage():
    """Any scorer's mean localization success is <= E[R_tau], for many scorers."""
    rng = np.random.default_rng(2)
    tau = 0.5
    overlaps_per_input = [rng.uniform(0.0, 1.0, size=int(rng.integers(3, 8))) for _ in range(200)]
    ceiling = expected_coverage(overlaps_per_input, tau)
    for _ in range(50):  # 50 arbitrary scorers, including an adversarial one
        chosen = [int(rng.integers(0, len(ov))) for ov in overlaps_per_input]
        assert coverage_ceiling_holds(overlaps_per_input, chosen, tau)
        succ = np.mean([localization_success(ov, c, tau) for ov, c in zip(overlaps_per_input, chosen)])
        assert succ <= ceiling + 1e-9
    # An oracle scorer that always picks the best-overlap proposal *attains* E[R_tau].
    oracle = [int(np.argmax(ov)) for ov in overlaps_per_input]
    oracle_succ = np.mean([localization_success(ov, c, tau) for ov, c in zip(overlaps_per_input, oracle)])
    assert abs(oracle_succ - ceiling) < 1e-9


# --------------------------------------------------------------------------- #
# Piece 4: repair-localization conflict
# --------------------------------------------------------------------------- #
def test_repair_localization_conflict():
    ex = repair_localization_conflict_example()
    loc = ex.localization_optimal()
    rep = ex.repair_optimal()
    # The localization-optimal proposal is the high-text-overlap one...
    assert loc.name == "text_box"
    # ...but it does NOT repair (a competitor still wins).
    assert not repairs(loc.repaired_logits, ex.y_star)
    # The repair-optimal proposal is a low-overlap non-text region that DOES repair.
    assert rep.name == "object_region"
    assert repairs(rep.repaired_logits, ex.y_star)
    # Hence the repair-optimal proposal is NOT the localization-optimal proposal.
    assert rep.name != loc.name
    assert rep.text_overlap < loc.text_overlap
