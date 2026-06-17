from __future__ import annotations

"""Internal PyTorch LoRA for the Real-OpenCLIP CSA pilot (no PEFT dependency).

This module implements *actual* Low-Rank Adaptation (LoRA) by patching selected
``nn.Linear`` modules inside the OpenCLIP **visual tower**. It deliberately does
**not** use the PEFT library: the previous pilot run could not import PEFT and
therefore fell back to an ``adapter_head_only`` module over frozen image
embeddings (which is *not* LoRA). The :class:`LoRALinear` wrapper here adapts the
backbone weights directly, so the ``manual_lora_visual`` mode is real LoRA.

Bounded framing
---------------
This is tooling for a *bounded CSA-LoRA pilot* on a controlled text-overlay
shortcut. It is **not** universal robustness, **not** open-world shortcut
discovery, **not** an RLHF/DPO replacement, **not** deployment validation, and
**not** a replacement for the finalized STS report.

Design notes
------------
* :class:`LoRALinear` wraps a frozen ``nn.Linear``. The original weight/bias are
  frozen; trainable low-rank matrices ``A`` and ``B`` are added so the layer
  computes ``y = W x + b + (alpha / r) * B A x``. ``A`` is initialised randomly
  and ``B`` is initialised to **zero**, so at initialisation the wrapped layer
  reproduces the frozen layer exactly.
* The wrapper also exposes merged ``weight``/``bias`` *properties*. This is what
  lets LoRA work on the attention output projection: ``nn.MultiheadAttention``
  reads ``out_proj.weight`` / ``out_proj.bias`` directly through the functional
  attention path rather than *calling* the module, so a wrapper that only
  overrode ``forward`` would be a silent no-op there. The merged-weight property
  keeps the low-rank delta (and its gradient to ``A``/``B``) in the attention
  computation while the MLP linears use the standard ``forward`` path.
"""

from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default LoRA targets in an OpenCLIP ``ResidualAttentionBlock``:
#   * ``c_fc`` / ``c_proj`` — the MLP linears (called as modules in forward).
#   * ``out_proj``          — the attention output projection (merged-weight path).
DEFAULT_TARGET_MODULES = ("c_fc", "c_proj", "out_proj")


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around a frozen ``nn.Linear``.

    ``y = W x + b + (alpha / r) * B (A x)``

    The wrapped linear's weight and bias are frozen. ``A`` (shape ``[r, in]``) is
    initialised randomly; ``B`` (shape ``[out, r]``) is initialised to zero, so an
    untrained wrapper matches the frozen layer exactly (to numerical tolerance).
    Dropout is supported on the LoRA branch (default ``0.0``).
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects an nn.Linear, got {type(base).__name__}")
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        # Trainable low-rank factors: A random, B zero -> initial delta is zero.
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.reset_lora_parameters()

        self.lora_dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def reset_lora_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """Re-initialise the low-rank factors (A random, B zero)."""

        with torch.no_grad():
            if generator is None:
                nn.init.normal_(self.lora_A, std=1.0 / self.rank)
            else:
                self.lora_A.copy_(
                    torch.randn(self.lora_A.shape, generator=generator) / self.rank
                )
            self.lora_B.zero_()

    @property
    def lora_delta(self) -> torch.Tensor:
        """The low-rank weight delta ``(alpha / r) * B A`` (shape ``[out, in]``)."""

        return self.scaling * (self.lora_B @ self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Merged effective weight ``W + (alpha / r) B A``.

        Used when a parent module (e.g. ``nn.MultiheadAttention``) reads the
        linear's ``weight`` directly instead of calling ``forward``.
        """

        return self.base.weight + self.lora_delta

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora = F.linear(self.lora_dropout(x), self.lora_A)  # [..., r]
        lora = F.linear(lora, self.lora_B)  # [..., out]
        return result + self.scaling * lora

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"


# --------------------------------------------------------------------------- #
# Module-graph helpers
# --------------------------------------------------------------------------- #
def get_visual_blocks(model: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the visual-tower transformer block list.

    Handles the standard OpenCLIP ``model.visual.transformer.resblocks`` layout
    and falls back to the first non-empty ``nn.ModuleList`` found under
    ``model.visual``. Returns ``None`` if no block list can be located.
    """

    visual = getattr(model, "visual", None)
    if visual is None:
        return None
    transformer = getattr(visual, "transformer", None)
    if transformer is not None:
        blocks = getattr(transformer, "resblocks", None)
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            return blocks
    for _name, sub in visual.named_modules():
        if isinstance(sub, nn.ModuleList) and len(sub) > 0:
            return sub
    return None


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace ``root.<dotted_name>`` with ``new_module``."""

    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _iter_target_linears(block: nn.Module, target_modules: Iterable[str]):
    """Yield ``(dotted_name, linear)`` for matching ``nn.Linear`` submodules.

    A linear matches if the *last* component of its dotted name is in
    ``target_modules``. ``LoRALinear`` instances are skipped (idempotent).
    """

    targets = tuple(target_modules)
    for name, sub in block.named_modules():
        if isinstance(sub, LoRALinear):
            continue
        if isinstance(sub, nn.Linear) and name.split(".")[-1] in targets:
            yield name, sub


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """Freeze every parameter that is not a LoRA factor."""

    lora_param_ids = set()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_param_ids.add(id(module.lora_A))
            lora_param_ids.add(id(module.lora_B))
    for p in model.parameters():
        p.requires_grad_(id(p) in lora_param_ids)


def reset_all_lora_parameters(model: nn.Module, seed: Optional[int] = None) -> None:
    """Re-initialise every :class:`LoRALinear` in ``model`` (A random, B zero)."""

    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(int(seed))
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_lora_parameters(generator=gen)


def lora_module_names(model: nn.Module) -> list[str]:
    return [name for name, sub in model.named_modules() if isinstance(sub, LoRALinear)]


# --------------------------------------------------------------------------- #
# Patching
# --------------------------------------------------------------------------- #
def apply_lora_to_openclip_visual(
    model: nn.Module,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    max_layers: Optional[int] = None,
) -> dict[str, Any]:
    """Patch selected ``nn.Linear`` modules in the OpenCLIP visual tower with LoRA.

    Only the **last** ``max_layers`` transformer blocks of the visual encoder are
    patched (all blocks if ``max_layers`` is ``None``). Within each selected block
    every ``nn.Linear`` whose name ends in one of ``target_modules`` is wrapped in
    :class:`LoRALinear`. The text tower is never touched. All non-LoRA parameters
    are frozen.

    Returns a record with the trainable-parameter count, the list of patched
    module names (fully-qualified from ``model``), and metadata. Raises
    ``ValueError`` if **zero** target modules are found.
    """

    blocks = get_visual_blocks(model)
    if blocks is None:
        raise ValueError(
            "could not locate a visual-tower transformer block list "
            "(expected model.visual.transformer.resblocks or a ModuleList under model.visual)"
        )

    n_blocks = len(blocks)
    if max_layers is None or max_layers >= n_blocks:
        selected_indices = list(range(n_blocks))
    else:
        selected_indices = list(range(n_blocks - int(max_layers), n_blocks))

    patched_names: list[str] = []
    for block_idx in selected_indices:
        block = blocks[block_idx]
        # Materialise the matches first so we don't mutate during iteration.
        matches = list(_iter_target_linears(block, target_modules))
        for local_name, linear in matches:
            wrapped = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
            _set_submodule(block, local_name, wrapped)
            patched_names.append(f"visual.transformer.resblocks.{block_idx}.{local_name}")

    if not patched_names:
        raise ValueError(
            "LoRA patching found zero target modules in the visual tower "
            f"(target_modules={tuple(target_modules)}, selected blocks={selected_indices}). "
            "Check target module names and the visual-tower structure."
        )

    freeze_non_lora_parameters(model)
    trainable = count_trainable_parameters(model)

    return {
        "patched_modules": patched_names,
        "num_patched_modules": len(patched_names),
        "trainable_param_count": int(trainable),
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "num_visual_blocks": int(n_blocks),
        "patched_block_indices": selected_indices,
        "target_modules": list(target_modules),
        "lora_used": True,
    }


__all__ = [
    "DEFAULT_TARGET_MODULES",
    "LoRALinear",
    "apply_lora_to_openclip_visual",
    "get_visual_blocks",
    "count_trainable_parameters",
    "freeze_non_lora_parameters",
    "reset_all_lora_parameters",
    "lora_module_names",
]
