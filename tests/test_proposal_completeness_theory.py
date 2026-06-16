from __future__ import annotations

"""Consistency checks for docs/global_cic_theory_attempt.md.

These tests confirm the proved guarantees hold exactly, and — just as importantly —
that they *fail* once an assumption is dropped. The impossibility (no-free-lunch)
test exhibits a concrete numeric witness of two worlds that are observationally
identical under a finite query set yet require different repair actions.
"""

import numpy as np

from causal_reliability.theory.proposal_completeness import (
    Candidate,
    argmaxes_agree,
    class_imbalance,
    gate_passes,
    gated_repair_certified,
    is_good_intervention,
    logit_margin,
    recovery_certified_by_clean_margin,
    recovery_certified_by_repaired_margin,
    selection_lemma_holds,
    top1_selects_good,
)


# --------------------------------------------------------------------------- #
# Lemma 2: per-input residual-to-clean recovery
# --------------------------------------------------------------------------- #
def test_class_independent_offset_never_changes_argmax():
    clean = np.array([3.0, 1.0, 0.0])
    # Pure class-independent offset (the "rho_bar" term): cannot change argmax.
    repaired = clean + 5.0
    assert class_imbalance(repaired - clean) < 1e-9
    assert argmaxes_agree(repaired, clean)


def test_recovery_holds_when_margin_exceeds_2epsB():
    rng = np.random.default_rng(0)
    eps_B = 0.5
    successes = 0
    trials = 500
    for _ in range(trials):
        clean = rng.normal(size=4)
        # Force a clean margin > 2 eps_B by separating the top class.
        clean[0] = clean.max() + 2.0 * eps_B + 0.3
        offset = rng.normal() * 3.0
        residual = rng.uniform(-eps_B, eps_B, size=4)  # |r_y| <= eps_B
        repaired = clean + offset + residual
        if recovery_certified_by_clean_margin(repaired, clean, eps_B):
            # Whenever the sufficient condition is certified, argmaxes MUST agree.
            assert argmaxes_agree(repaired, clean)
            successes += 1
    assert successes > 0  # the condition is actually exercised


def test_recovery_can_fail_when_margin_too_small():
    # Clean margin < 2 eps_B: a worst-case residual flips the argmax.
    eps_B = 1.0
    clean = np.array([0.5, 0.0, 0.0])  # margin 0.5 < 2*eps_B
    residual = np.array([-eps_B, +eps_B, 0.0])  # legal |r_y| <= eps_B
    repaired = clean + residual
    assert is_good_intervention(repaired, clean, eps_B)
    assert not recovery_certified_by_clean_margin(repaired, clean, eps_B)
    assert not argmaxes_agree(repaired, clean)  # argmax actually flipped


# --------------------------------------------------------------------------- #
# Lemma 3: observable-margin recovery
# --------------------------------------------------------------------------- #
def test_observable_repaired_margin_implies_clean_argmax():
    rng = np.random.default_rng(1)
    eps_B = 0.4
    for _ in range(500):
        clean = rng.normal(size=5)
        offset = rng.normal() * 2.0
        residual = rng.uniform(-eps_B, eps_B, size=5)
        repaired = clean + offset + residual
        if recovery_certified_by_repaired_margin(repaired, eps_B):
            # m(T_a(x)) > 2 eps_B  =>  repaired winner == clean winner.
            assert argmaxes_agree(repaired, clean)


# --------------------------------------------------------------------------- #
# Lemma 1: selection
# --------------------------------------------------------------------------- #
def test_selection_lemma_top1_is_good_under_margin_and_low_noise():
    rng = np.random.default_rng(2)
    candidates = [
        Candidate("good", noiseless_score=1.0, is_good=True),
        Candidate("harm1", noiseless_score=0.4, is_good=False),
        Candidate("harm2", noiseless_score=0.2, is_good=False),
    ]
    gamma = 0.6  # good leads every harmful by >= 0.6
    noise_bound = gamma / 2.0
    assert selection_lemma_holds(candidates, gamma, noise_bound)
    for _ in range(500):
        observed = {c.name: c.noiseless_score + rng.uniform(-noise_bound, noise_bound) for c in candidates}
        assert top1_selects_good(candidates, observed)


def test_selection_can_fail_when_noise_exceeds_half_gamma():
    candidates = [
        Candidate("good", noiseless_score=1.0, is_good=True),
        Candidate("harm", noiseless_score=0.4, is_good=False),
    ]
    gamma = 0.6
    big_noise = 0.5  # > gamma/2 = 0.3
    assert not selection_lemma_holds(candidates, gamma, big_noise)
    # A concrete adversarial noise draw makes the harmful candidate win.
    observed = {"good": 1.0 - big_noise, "harm": 0.4 + big_noise}  # 0.5 vs 0.9
    assert not top1_selects_good(candidates, observed)


# --------------------------------------------------------------------------- #
# Theorem 1: composition (selection + recovery)
# --------------------------------------------------------------------------- #
def test_theorem1_composition_repairs_to_clean_prediction():
    eps_B = 0.3
    clean = np.array([2.0, 0.0, 0.0])  # margin 2.0 > 2*eps_B
    good_repaired = clean + 1.0 + np.array([0.1, -0.1, 0.0])  # good intervention
    harmful_repaired = np.array([0.0, 2.0, 0.0])  # flips prediction to class 1
    candidates = [
        Candidate("good", noiseless_score=1.0, is_good=True),
        Candidate("harm", noiseless_score=0.3, is_good=False),
    ]
    gamma, noise = 0.7, 0.3
    assert selection_lemma_holds(candidates, gamma, noise)
    observed = {"good": 1.0, "harm": 0.3}
    selected_repaired = good_repaired if top1_selects_good(candidates, observed) else harmful_repaired
    assert recovery_certified_by_clean_margin(selected_repaired, clean, eps_B)
    assert argmaxes_agree(selected_repaired, clean)
    assert int(np.argmax(selected_repaired)) == 0  # the clean causal class


# --------------------------------------------------------------------------- #
# Theorem 2: observable gate
# --------------------------------------------------------------------------- #
def test_gate_certificate():
    thresholds = {
        "min_score_gap": 0.2,
        "min_repaired_margin": 1.0,
        "min_stability_gain": 0.1,
        "max_area_fraction": 0.5,
        "min_clean_safe_score": 0.7,
    }
    eps_hat = 0.4
    repaired = np.array([2.0, 0.0, 0.0])  # margin 2.0 > 2*eps_hat
    ok = gate_passes(
        score_gap_top1_median=0.5,
        repaired_margin=2.0,
        stability_gain=0.3,
        selected_area_fraction=0.3,
        clean_safe_score=0.9,
        thresholds=thresholds,
    )
    assert ok
    assert gated_repair_certified(gate_ok=ok, repaired_logits=repaired, eps_hat=eps_hat)

    # Gate fails (area too large) -> no certificate even if margin is large.
    bad = gate_passes(
        score_gap_top1_median=0.5,
        repaired_margin=2.0,
        stability_gain=0.3,
        selected_area_fraction=0.9,
        clean_safe_score=0.9,
        thresholds=thresholds,
    )
    assert not bad
    assert not gated_repair_certified(gate_ok=bad, repaired_logits=repaired, eps_hat=eps_hat)


# --------------------------------------------------------------------------- #
# Theorem 3: no-free-lunch / impossibility (numeric witness)
# --------------------------------------------------------------------------- #
def test_impossibility_two_worlds_indistinguishable_under_finite_queries():
    """Two worlds agree on every observable, but require different repairs.

    Query set Q = {a1, a2}. The algorithm observes the logits on x and on the
    queried interventions T_a1(x), T_a2(x) -- and these are *identical* across the
    two worlds. What differs is the unobservable causal label y* and which
    intervention is the true neutralizer. A finite-query black-box algorithm must
    pick the same action in both worlds; that action repairs in W1 but not in W2.
    This is the failure of proposal completeness made concrete.
    """

    # Observations available to the algorithm (identical in both worlds):
    obs_x = np.array([0.0, 0.1, 0.0])
    obs_after_a1 = np.array([0.2, 0.0, 0.0])  # argmax 0
    obs_after_a2 = np.array([0.0, 0.0, 0.2])  # argmax 2

    # World 1: the causal label is y* = 0. Applying a1 yields argmax 0 == y*.
    w1_ystar = 0
    # World 2: the causal label is y* = 2 (UNOBSERVABLE difference). The only true
    # neutralizer is a_star, which is NOT in the finite query set Q.
    w2_ystar = 2
    w2_after_astar = np.array([0.0, 0.0, 1.5])  # restores class 2, never queried

    # The observations on the finite query set are identical across the two worlds,
    # so any black-box algorithm restricted to Q behaves identically in both.
    chosen_action = "a1"  # whatever it picks, it is the same in both worlds
    assert chosen_action in {"a1", "a2"}  # a_star is outside its proposal/query set

    chosen_logits = obs_after_a1 if chosen_action == "a1" else obs_after_a2
    # Same action: its repaired prediction is the same vector in both worlds...
    repaired_pred = int(np.argmax(chosen_logits))
    # ...but it matches y* in W1 and not in W2.
    assert repaired_pred == w1_ystar  # W1: action repairs
    assert repaired_pred != w2_ystar  # W2: same action fails
    # The true repair in W2 exists but is outside the finite query/proposal set.
    assert int(np.argmax(w2_after_astar)) == w2_ystar


def test_completeness_rules_out_the_bad_world():
    """If the family IS complete (contains a good intervention) and CIC ranks it,
    the bad world cannot arise — Theorem 1 applies."""

    eps_B = 0.3
    clean = np.array([1.5, 0.0, 0.0])  # margin 1.5 > 2*eps_B
    a_star_repaired = clean + 0.5 + np.array([0.1, -0.1, 0.0])  # good intervention in-family
    assert is_good_intervention(a_star_repaired, clean, eps_B)
    assert recovery_certified_by_clean_margin(a_star_repaired, clean, eps_B)
    assert argmaxes_agree(a_star_repaired, clean)
