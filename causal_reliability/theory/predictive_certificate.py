from __future__ import annotations

"""Machine-checkable encoding of the **conditional predictive certificate** that
underlies the practical predictive CIC reliability gate.

This is **not** a universal theorem and makes **no** claim of universal correctness.
It is a *label-free abstention rule*: it says when an observed repaired prediction
is stable under the calibrated perturbation class, which is the property the
empirical gate (``causal_reliability.analysis.predictive_cic_gate``) is calibrated
to track.

It builds directly on the proposal-complete recovery lemmas in
``causal_reliability.theory.proposal_completeness`` (in particular
``recovery_certified_by_repaired_margin``: if an intervention's residual-to-clean
class imbalance is at most ``eps_B`` and the observed repaired logit margin exceeds
``2 eps_B``, the repaired argmax is stable). Here ``eps_B`` is replaced by an
**empirically calibrated** upper bound ``eps_hat`` on the residual instability of
the calibrated perturbation class, estimated from validation data with NO test
label.

Proposition (predictive CIC certificate, conditional).
    Let ``a`` be the CIC-selected intervention for input ``x`` and let
    ``m_rep(x) = top1 - top2`` be the *observed* margin of the repaired logits
    ``ell(T_a(x))``. Let ``eps_hat`` be a margin-quantile-calibrated upper bound on
    the residual-to-clean class imbalance of the calibrated perturbation class
    ``A_cal`` (Section: per-input class balance). If

        m_rep(x) > 2 * eps_hat,

    then the repaired prediction is **stable** under ``A_cal``: every perturbation
    ``a' in A_cal`` whose residual class imbalance is at most ``eps_hat`` leaves the
    repaired argmax unchanged. This certifies *stability under the calibrated
    perturbation class*, which is observable and label-free; it does NOT certify
    universal correctness. Below the threshold the rule abstains.

This module introduces no experimental metric and is imported by no experiment
runner; it operates on logit vectors and scalar bounds only and exists so the unit
tests can confirm the certificate fires exactly when its premise holds and abstains
otherwise (including a numeric witness that the bound is *necessary*).
"""

from dataclasses import dataclass

import numpy as np

from causal_reliability.theory.proposal_completeness import (
    class_imbalance,
    logit_margin,
    recovery_certified_by_repaired_margin,
    residual_to_clean,
)

ArrayLike = np.ndarray


def calibrate_residual_bound(
    residual_imbalances: ArrayLike, quantile: float = 0.9
) -> float:
    """Calibrate ``eps_hat`` as an upper quantile of the observed residual-to-clean
    class imbalances of the calibrated perturbation class on validation inputs.

    Using an upper quantile (not the max) yields a *conservative-but-not-paranoid*
    bound: the certificate then holds for the modelled ``A_cal`` whose per-input
    imbalance is at most ``eps_hat``. Returns ``+inf`` for an empty sample so the
    certificate cannot fire without calibration data.
    """

    vals = np.asarray([v for v in np.asarray(residual_imbalances, dtype=float) if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("inf")
    q = min(max(quantile, 0.0), 1.0)
    return float(np.quantile(vals, q))


def certified_stable(repaired_logits: ArrayLike, eps_hat: float) -> bool:
    """The conditional certificate: ``m_rep > 2 eps_hat`` ==> stable under ``A_cal``.

    This is exactly ``recovery_certified_by_repaired_margin`` with the *calibrated*
    bound ``eps_hat`` substituted for the residual budget. It is observable
    (depends only on the repaired logits and the calibrated scalar) and label-free.
    """

    if not np.isfinite(eps_hat):
        return False
    return recovery_certified_by_repaired_margin(repaired_logits, eps_hat)


def stable_under_perturbation(
    repaired_logits: ArrayLike, perturbations: list[ArrayLike], eps_hat: float
) -> bool:
    """Ground-truth stability witness used by the tests: whether every supplied
    perturbed logit vector whose imbalance is within ``eps_hat`` keeps the argmax.

    ``perturbations`` are *deltas* added to ``repaired_logits``. A perturbation with
    imbalance above ``eps_hat`` is outside ``A_cal`` and is ignored.
    """

    base = np.asarray(repaired_logits, dtype=float)
    base_arg = int(np.argmax(base))
    for delta in perturbations:
        d = np.asarray(delta, dtype=float)
        if class_imbalance(d) > eps_hat + 1e-12:
            continue
        if int(np.argmax(base + d)) != base_arg:
            return False
    return True


@dataclass(frozen=True)
class PredictiveCertificate:
    """Bundles a calibrated bound with the decision for one input."""

    eps_hat: float
    repaired_margin: float

    @property
    def fires(self) -> bool:
        return bool(np.isfinite(self.eps_hat) and self.repaired_margin > 2.0 * self.eps_hat)

    @property
    def decision(self) -> str:
        return "certify_stable" if self.fires else "abstain"


def certificate_for(repaired_logits: ArrayLike, eps_hat: float) -> PredictiveCertificate:
    return PredictiveCertificate(eps_hat=float(eps_hat), repaired_margin=logit_margin(repaired_logits))


__all__ = [
    "calibrate_residual_bound",
    "certified_stable",
    "stable_under_perturbation",
    "PredictiveCertificate",
    "certificate_for",
    "residual_to_clean",
]
