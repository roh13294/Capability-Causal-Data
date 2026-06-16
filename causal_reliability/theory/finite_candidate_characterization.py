from __future__ import annotations

"""Machine-checkable encoding of the **complete finite-candidate characterization
of CIC repair**.

This module is a *consistency check* of the four theorem-package statements added to
``docs/theory.md`` (Section 11, "Complete finite-candidate characterization of CIC
repair") and summarized in ``paper/main.tex``. It implements the exact inequalities
so the unit tests can confirm the guarantees hold exactly as stated, and — just as
importantly — that they are *tight* (the certificate flips at its stated boundary).

Scope, stated honestly. This is a characterization **inside a finite candidate set**.
It does **not** claim open-world shortcut discovery, semantic correctness, exact
text-box localization, or general robustness. It introduces **no** experimental
metric, is imported by **no** experiment runner, changes **no** result JSON, and
operates on logit vectors / overlap scalars only.

Notation (matches the docs):

* ``A(x)``       : a finite proposal/intervention set indexed by candidate ``a``.
* ``T_a(x)``     : the repaired/intervened input for candidate ``a``.
* ``z_a``        : ``f(T_a(x))`` — repaired logits for candidate ``a``.
* ``z_clean``    : ``f(x_clean)`` — clean logits (decomposition ``z_a = z_clean + r_a``).
* ``r_a``        : residual ``z_a - z_clean``.
* ``y_star``     : the clean/causal label, used only for the theorem/evaluation,
                   never for inference.
"""

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray


# --------------------------------------------------------------------------- #
# Piece 1: Exact repair criterion (necessary and sufficient)
# --------------------------------------------------------------------------- #
def repairs(repaired_logits: ArrayLike, y_star: int) -> bool:
    """Exact repair criterion. Candidate ``a`` repairs iff

        z_a(y*) > max_{j != y*} z_a(j).

    This is **necessary and sufficient** for repair inside the finite candidate set:
    repair *is* the event that the repaired argmax equals ``y*`` (with strict
    separation from every competitor).
    """

    z = np.asarray(repaired_logits, dtype=float)
    ystar = int(y_star)
    competitors = np.delete(z, ystar)
    if competitors.size == 0:
        return True
    return bool(z[ystar] > float(competitors.max()))


def repairs_via_residual(clean_logits: ArrayLike, residual: ArrayLike, y_star: int) -> bool:
    """Equivalent residual form. With ``z_a = z_clean + r_a``, repair occurs iff for
    every competing label ``j``:

        z_clean(y*) - z_clean(j) > r_a(j) - r_a(y*).

    Equality with :func:`repairs` is checked by the tests for arbitrary inputs.
    """

    z_clean = np.asarray(clean_logits, dtype=float)
    r = np.asarray(residual, dtype=float)
    ystar = int(y_star)
    for j in range(z_clean.size):
        if j == ystar:
            continue
        if not (z_clean[ystar] - z_clean[j] > r[j] - r[ystar]):
            return False
    return True


# --------------------------------------------------------------------------- #
# Piece 2: Tight residual-margin certificate
# --------------------------------------------------------------------------- #
def class_imbalance(residual: ArrayLike) -> float:
    """``max_y |r_y - mean_y r|`` — the per-input class-centered imbalance of a
    residual. The certificate bound ``epsilon`` is a bound on this quantity."""

    r = np.asarray(residual, dtype=float)
    return float(np.max(np.abs(r - r.mean())))


def causal_margin(clean_logits: ArrayLike, y_star: int) -> float:
    """``m_clean = z_clean(y*) - max_{j != y*} z_clean(j)``."""

    z = np.asarray(clean_logits, dtype=float)
    ystar = int(y_star)
    competitors = np.delete(z, ystar)
    if competitors.size == 0:
        return float("inf")
    return float(z[ystar] - competitors.max())


def margin_certificate_sufficient(clean_logits: ArrayLike, y_star: int, eps: float) -> bool:
    """Sufficiency half of the certificate: if the residual class-score imbalance is
    bounded by ``eps`` then ``m_clean > 2 eps`` is sufficient for repair stability.

    Returns whether the sufficient condition ``m_clean > 2 eps`` holds; when it does,
    *every* residual with imbalance ``<= eps`` repairs (verified by the tests)."""

    return bool(causal_margin(clean_logits, y_star) > 2.0 * eps + 0.0)


def tightness_witness(clean_logits: ArrayLike, y_star: int, eps: float) -> np.ndarray:
    """Tightness half of the certificate. If the clean/causal margin is at most
    ``2 eps``, this returns an *allowed* residual perturbation (class imbalance
    ``<= eps``) whose worst-case pairwise swing against ``y*`` is exactly ``2 eps``,
    so it flips the top class.

    Construction: subtract ``eps`` from ``y*`` and add ``eps`` to the leading
    competitor; the two offsets cancel so the residual is mean-zero and its
    class-centered imbalance is exactly ``eps``. The pairwise swing
    ``r(j) - r(y*) = 2 eps`` then meets or exceeds the margin, so the competitor
    catches or overtakes ``y*``. The certificate is therefore sharp: no bound looser
    than ``2 eps`` is possible under the residual-instability model.
    """

    z = np.asarray(clean_logits, dtype=float)
    ystar = int(y_star)
    competitors = [j for j in range(z.size) if j != ystar]
    # Leading competitor — the one that defines the margin.
    j = max(competitors, key=lambda k: z[k])
    r = np.zeros_like(z)
    r[ystar] = -eps
    r[j] = +eps
    return r


# --------------------------------------------------------------------------- #
# Piece 3: Proposal coverage ceiling
# --------------------------------------------------------------------------- #
def coverage_indicator(overlaps: ArrayLike, tau: float) -> int:
    """``R_tau(x) = 1[ exists a candidate proposal a in A(x) with overlap >= tau ]``."""

    ov = np.asarray(overlaps, dtype=float)
    if ov.size == 0:
        return 0
    return int(bool(np.any(ov >= tau)))


def localization_success(overlaps: ArrayLike, chosen_idx: int, tau: float) -> int:
    """Localization success at threshold ``tau`` for a scorer that selected
    ``chosen_idx`` from ``A(x)``: ``1[ overlap(chosen) >= tau ]``."""

    ov = np.asarray(overlaps, dtype=float)
    return int(bool(ov[int(chosen_idx)] >= tau))


def coverage_ceiling_holds(
    overlaps_per_input: list[ArrayLike], chosen_idx_per_input: list[int], tau: float
) -> bool:
    """Proposal coverage ceiling theorem. For **any** scoring/ranking method that
    chooses one candidate per input from ``A(x)``, mean localization success at
    threshold ``tau`` is upper-bounded by ``E[R_tau(x)]``.

    Per input, ``localization_success <= R_tau`` (a chosen proposal can only overlap
    if some proposal does); averaging preserves the inequality. Returns whether the
    bound holds for the supplied (scorer, dataset).
    """

    succ = 0.0
    cover = 0.0
    n = len(overlaps_per_input)
    for ov, idx in zip(overlaps_per_input, chosen_idx_per_input):
        succ += localization_success(ov, idx, tau)
        cover += coverage_indicator(ov, tau)
    # mean success <= mean coverage (allow tiny float slack)
    return bool(succ <= cover + 1e-9)


def expected_coverage(overlaps_per_input: list[ArrayLike], tau: float) -> float:
    """``E[R_tau(x)]`` over the dataset — the ceiling no scorer can exceed."""

    if not overlaps_per_input:
        return 0.0
    return float(np.mean([coverage_indicator(ov, tau) for ov in overlaps_per_input]))


# --------------------------------------------------------------------------- #
# Piece 4: Repair-localization conflict
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConflictCandidate:
    """One candidate in a finite-candidate repair problem."""

    name: str
    text_overlap: float            # overlap with the human/text annotation box
    repaired_logits: np.ndarray    # z_a = f(T_a(x))


@dataclass(frozen=True)
class ConflictExample:
    y_star: int
    candidates: list[ConflictCandidate]

    def localization_optimal(self) -> ConflictCandidate:
        """The proposal that maximizes human-box (text) overlap."""

        return max(self.candidates, key=lambda c: c.text_overlap)

    def repair_optimal(self) -> ConflictCandidate:
        """The proposal that best restores the causal label — first by whether it
        repairs (exact criterion), then by repaired margin for ``y_star``."""

        def key(c: ConflictCandidate) -> tuple[int, float]:
            z = np.asarray(c.repaired_logits, dtype=float)
            comp = np.delete(z, self.y_star)
            margin = float(z[self.y_star] - comp.max()) if comp.size else float("inf")
            return (int(repairs(c.repaired_logits, self.y_star)), margin)

        return max(self.candidates, key=key)


def repair_localization_conflict_example() -> ConflictExample:
    """A concrete finite-candidate repair problem in which the **repair-optimal**
    proposal is **not** the **localization-optimal** proposal.

    * ``text_box`` — high human-box (text) overlap, but a *weak* repair effect: it
      does not restore the causal label ``y* = 0`` (a competing class still wins).
    * ``object_region`` — low text overlap, but a *stronger* repair effect that does
      restore ``y* = 0``.

    This witnesses the repair-localization conflict theorem: repair success does not
    imply text-box localization.
    """

    y_star = 0
    text_box = ConflictCandidate(
        name="text_box",
        text_overlap=0.90,
        # Competing class 1 still wins after neutralizing the literal text box:
        repaired_logits=np.array([0.20, 0.80, 0.00]),
    )
    object_region = ConflictCandidate(
        name="object_region",
        text_overlap=0.05,
        # Causal class 0 restored after neutralizing a non-text region:
        repaired_logits=np.array([1.50, 0.30, 0.00]),
    )
    return ConflictExample(y_star=y_star, candidates=[text_box, object_region])


__all__ = [
    "repairs",
    "repairs_via_residual",
    "class_imbalance",
    "causal_margin",
    "margin_certificate_sufficient",
    "tightness_witness",
    "coverage_indicator",
    "localization_success",
    "coverage_ceiling_holds",
    "expected_coverage",
    "ConflictCandidate",
    "ConflictExample",
    "repair_localization_conflict_example",
]
