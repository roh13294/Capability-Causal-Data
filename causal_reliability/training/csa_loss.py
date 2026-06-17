from __future__ import annotations

"""Loss functions for the Counterfactual Stability Alignment (CSA) pilot.

CSA is a *pilot* that asks a single, bounded question: can the test-time CIC
counterfactual-instability signal be re-used as a small **training-time
regularizer** that reduces shortcut reliance under finite-candidate
interventions while preserving clean accuracy?

These losses operate on classifier ``logits`` (a lightweight head on top of
frozen embeddings — see :mod:`causal_reliability.training.csa_trainer`). They are
deliberately simple and differentiable so a tiny head can be trained in a few
steps. The four building blocks mirror the four CSA terms:

1. ``js_divergence_logits``  — symmetric, bounded divergence between two predictive
   distributions; used as a stability primitive.
2. ``kl_stability_loss``     — predictions on the *clean* image should not move
   when a shortcut intervention is applied.
3. ``cic_instability_penalty`` — predictions should be *stable across* the finite
   set of candidate interventions (this is the CIC signal, used as a loss).
4. ``clean_preservation_loss`` — clean predictions should stay anchored to a
   reference (a frozen reader / the pre-training prediction) so clean accuracy
   does not collapse.

``csa_total_loss`` combines a task cross-entropy with the three regularizers.

Scope / non-claims: this is a regularizer pilot, not an RLHF/DPO replacement, not
universal robustness, and not open-world shortcut discovery.
"""

from typing import Optional

import torch
import torch.nn.functional as F


_EPS = 1e-8


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")


def js_divergence_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    *,
    eps: float = _EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """Jensen-Shannon divergence between ``softmax(logits_p)`` and ``softmax(logits_q)``.

    Symmetric and bounded in ``[0, ln 2]``. Exactly ``0`` for identical logits,
    which makes it a clean stability primitive: the two tensors must broadcast to
    the same shape and the divergence is computed over the last (class) axis.
    """

    p = F.softmax(logits_p, dim=-1).clamp_min(eps)
    q = F.softmax(logits_q, dim=-1).clamp_min(eps)
    m = (0.5 * (p + q)).clamp_min(eps)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return _reduce(js, reduction)


def kl_stability_loss(
    clean_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    *,
    eps: float = _EPS,
    symmetric: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize prediction drift between the clean image and shortcut interventions.

    ``clean_logits`` has shape ``[N, K]``. ``intervened_logits`` is either
    ``[N, K]`` (a single intervention) or ``[N, M, K]`` (``M`` finite candidate
    interventions); in the latter case the per-example loss is averaged over the
    ``M`` interventions. Returns ``KL(clean || intervened)`` (or the symmetric
    average when ``symmetric=True``). Zero iff the distributions match.
    """

    clean = clean_logits
    if intervened_logits.dim() == clean_logits.dim() + 1:
        clean = clean_logits.unsqueeze(1)  # [N, 1, K] broadcasts over the M axis
    logp_clean = F.log_softmax(clean, dim=-1)
    logp_int = F.log_softmax(intervened_logits, dim=-1)
    p_clean = logp_clean.exp()
    kl = (p_clean * (logp_clean - logp_int)).sum(dim=-1)
    if symmetric:
        p_int = logp_int.exp()
        kl = 0.5 * kl + 0.5 * (p_int * (logp_int - logp_clean)).sum(dim=-1)
    if kl.dim() > clean_logits.dim() - 1:
        kl = kl.mean(dim=1)  # average over the candidate axis -> [N]
    kl = kl.clamp_min(0.0)
    return _reduce(kl, reduction)


def cic_instability_penalty(
    candidate_logits: torch.Tensor,
    *,
    eps: float = _EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """Counterfactual-instability penalty across a finite candidate set.

    ``candidate_logits`` has shape ``[N, M, K]`` for ``M`` finite candidate
    interventions per example. The penalty is the *information radius* of the
    candidate predictions: the mean KL divergence from each candidate
    distribution to the candidate-mean distribution (a generalized
    Jensen-Shannon divergence). It is ``0`` iff all candidate predictions are
    identical — i.e. the prediction is counterfactually stable across the finite
    candidate set — and grows as predictions disagree.
    """

    if candidate_logits.dim() != 3:
        raise ValueError("candidate_logits must have shape [N, M, K]")
    p = F.softmax(candidate_logits, dim=-1).clamp_min(eps)  # [N, M, K]
    p_mean = p.mean(dim=1, keepdim=True).clamp_min(eps)  # [N, 1, K]
    per_candidate_kl = (p * (p / p_mean).log()).sum(dim=-1)  # [N, M]
    dispersion = per_candidate_kl.mean(dim=1).clamp_min(0.0)  # [N]
    return _reduce(dispersion, reduction)


def clean_preservation_loss(
    clean_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    mode: str = "kl",
    eps: float = _EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """Keep the clean prediction anchored to a (detached) reference reader.

    The reference is detached so it acts as a fixed target — typically the frozen
    reader's clean prediction or the head's pre-training clean prediction. With
    ``mode="kl"`` this is ``KL(reference || clean)``; with ``mode="mse"`` it is
    the mean-squared logit difference. Zero when the current clean prediction
    matches the reference, which prevents the regularizers from collapsing clean
    accuracy.
    """

    ref = reference_logits.detach()
    if mode == "kl":
        logp_ref = F.log_softmax(ref, dim=-1)
        logp_cur = F.log_softmax(clean_logits, dim=-1)
        p_ref = logp_ref.exp()
        loss = (p_ref * (logp_ref - logp_cur)).sum(dim=-1).clamp_min(0.0)
    elif mode == "mse":
        loss = ((clean_logits - ref) ** 2).mean(dim=-1)
    else:
        raise ValueError(f"unknown preservation mode: {mode!r}")
    return _reduce(loss, reduction)


def csa_total_loss(
    task_logits: torch.Tensor,
    labels: torch.Tensor,
    clean_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    intervened_logits: Optional[torch.Tensor] = None,
    reference_clean_logits: Optional[torch.Tensor] = None,
    lambda_stability: float = 1.0,
    lambda_cic: float = 1.0,
    lambda_preservation: float = 0.5,
    preservation_mode: str = "kl",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine the task loss with the three CSA regularizers.

    Components
    ----------
    * ``task``          — cross-entropy on the true object label (uses the
      observed/training image logits, which still contain the shortcut).
    * ``stability``     — :func:`kl_stability_loss` between clean and intervened
      predictions. If ``intervened_logits`` is ``None`` the full candidate set is
      used as the intervention bank.
    * ``cic``           — :func:`cic_instability_penalty` over the finite
      candidate set (the CIC signal turned into a training loss).
    * ``preservation``  — :func:`clean_preservation_loss` toward
      ``reference_clean_logits`` (skipped when ``None``).

    Returns ``(total_loss, components)`` where ``components`` maps each term name
    to its (pre-weight) scalar value plus the weighted ``total``.
    """

    task = F.cross_entropy(task_logits, labels)
    interv = intervened_logits if intervened_logits is not None else candidate_logits
    stability = kl_stability_loss(clean_logits, interv)
    cic = cic_instability_penalty(candidate_logits)
    if reference_clean_logits is not None:
        preservation = clean_preservation_loss(
            clean_logits, reference_clean_logits, mode=preservation_mode
        )
    else:
        preservation = clean_logits.new_zeros(())

    total = (
        task
        + lambda_stability * stability
        + lambda_cic * cic
        + lambda_preservation * preservation
    )
    components = {
        "task": float(task.detach()),
        "stability": float(stability.detach()),
        "cic": float(cic.detach()),
        "preservation": float(preservation.detach()),
        "total": float(total.detach()),
    }
    return total, components


__all__ = [
    "js_divergence_logits",
    "kl_stability_loss",
    "cic_instability_penalty",
    "clean_preservation_loss",
    "csa_total_loss",
]
