from __future__ import annotations

"""Tests for the internal PyTorch LoRA used by the ``manual_lora_visual`` mode.

These never load the real OpenCLIP backbone: a tiny OpenCLIP-*shaped* toy model
(``visual.transformer.resblocks`` with ``attn`` / ``mlp.c_fc`` / ``mlp.c_proj``)
exercises the patcher and a 1-step training loop on synthetic tensors. They
verify: LoRALinear is identity at init, patching freezes everything but the LoRA
factors, at least one visual-tower module is patched, the trainable count is
nonzero and small, zero-target patching errors clearly, and a tiny manual-LoRA
training step runs and updates only LoRA parameters.
"""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from causal_reliability.training import manual_lora as ml
from causal_reliability.training.manual_lora import (
    LoRALinear,
    apply_lora_to_openclip_visual,
    count_trainable_parameters,
    lora_module_names,
)
from causal_reliability.training.csa_lora import (
    LoraImageDataset,
    LoraImageSplit,
    LoraPilotConfig,
    ManualLoraModel,
    evaluate_manual_lora_mode,
    run_manual_lora_modes,
    train_manual_lora,
    compute_manual_lora_go_no_go,
)


# --------------------------------------------------------------------------- #
# Toy OpenCLIP-shaped model (no real weights loaded)
# --------------------------------------------------------------------------- #
class _ToyBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(dim, hidden)), ("gelu", nn.GELU()), ("c_proj", nn.Linear(hidden, dim))])
        )

    def forward(self, x):
        h = self.ln_1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x


class _ToyTransformer(nn.Module):
    def __init__(self, dim, hidden, depth):
        super().__init__()
        self.resblocks = nn.ModuleList([_ToyBlock(dim, hidden) for _ in range(depth)])

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class _ToyVisual(nn.Module):
    def __init__(self, dim, hidden, depth, patch):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=False)
        self.transformer = _ToyTransformer(dim, hidden, depth)
        self.ln_post = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv1(x)  # [B, dim, gh, gw]
        b, d, gh, gw = x.shape
        x = x.reshape(b, d, gh * gw).permute(0, 2, 1)  # [B, tokens, dim]
        x = self.transformer(x)
        return self.ln_post(x.mean(dim=1))


class ToyCLIP(nn.Module):
    """OpenCLIP-shaped toy: ``visual.transformer.resblocks`` + ``encode_image``."""

    def __init__(self, dim=16, hidden=32, depth=3, patch=4):
        super().__init__()
        self.visual = _ToyVisual(dim, hidden, depth, patch)
        self.embed_dim = dim

    def encode_image(self, x):
        return self.visual(x)


def _toy_manual_lora_model(dim=16, k=4, seed=0, patch_last=2):
    torch.manual_seed(seed)
    toy = ToyCLIP(dim=dim)
    info = apply_lora_to_openclip_visual(toy, rank=4, alpha=8.0, max_layers=patch_last)
    g = torch.Generator().manual_seed(seed + 1)
    text = torch.nn.functional.normalize(torch.randn(k, dim, generator=g), dim=-1)
    transfer = torch.nn.functional.normalize(torch.randn(k, dim, generator=g), dim=-1)
    model = ManualLoraModel(
        toy,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        logit_scale=10.0,
        text_features=text,
        transfer_text_features=transfer,
        encode_batch_size=8,
    )
    return model, info


def _tiny_image_dataset(n=6, k=4, img=8, seed=0):
    g = torch.Generator().manual_seed(seed)

    def split(m):
        rnd = lambda *shape: torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
        return LoraImageSplit(
            observed=rnd(m, 3, img, img),
            clean=rnd(m, 3, img, img),
            misleading=rnd(m, 3, img, img),
            candidates=rnd(m, k, 3, img, img),
            labels=torch.randint(0, k, (m,), generator=g),
        )

    return LoraImageDataset(
        train=split(n), val=split(n), held_out_val=split(n), transfer_val=split(n),
        shape_prompts=["a"] * k, decoy_prompts=["b"] * k,
        num_classes=k, num_transfer_classes=k, image_size=img,
    )


def _tiny_cfg(**overrides):
    cfg = LoraPilotConfig.from_dict({})
    cfg.manual_lora.epochs = 1
    cfg.manual_lora.batch_size = 4
    cfg.manual_lora.lr = 0.05
    cfg.manual_lora.encode_batch_size = 8
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# LoRALinear semantics
# --------------------------------------------------------------------------- #
def test_lora_linear_initial_identity():
    base = nn.Linear(12, 7)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(5, 12)
    # B initialised to zero => wrapped layer matches the frozen linear exactly.
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    # Merged-weight property (used by the MHA attention path) is also identity.
    assert torch.allclose(lora.weight, base.weight, atol=1e-6)
    assert torch.allclose(lora.bias, base.bias, atol=1e-6)


def test_lora_linear_factors_on_base_device():
    # The LoRA factors must be created on the SAME device as the frozen base
    # weight (CPU here), so the merged-weight property never mixes devices.
    base = nn.Linear(10, 6)
    lora = LoRALinear(base, rank=3, alpha=6.0)
    assert lora.lora_A.device == base.weight.device
    assert lora.lora_B.device == base.weight.device
    assert lora.lora_delta.device == base.weight.device
    assert lora.weight.device == base.weight.device
    assert lora.lora_A.dtype == base.weight.dtype
    assert lora.lora_B.dtype == base.weight.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lora_linear_cuda_weight_and_delta_on_cuda():
    # Regression: wrapping a CUDA nn.Linear must keep the LoRA factors, the
    # merged ``weight`` property, and the ``lora_delta`` on CUDA. Previously the
    # factors defaulted to CPU and ``weight = base.weight + lora_delta`` crashed
    # with "Expected all tensors to be on the same device" on the OpenCLIP
    # out_proj.weight attention path.
    base = nn.Linear(12, 7).cuda()
    assert base.weight.is_cuda
    lora = LoRALinear(base, rank=4, alpha=8.0)
    assert lora.lora_A.is_cuda
    assert lora.lora_B.is_cuda
    assert lora.lora_delta.is_cuda
    assert lora.weight.is_cuda
    # Still identity at init, now wholly on CUDA (B == 0 => weight == base.weight).
    assert torch.allclose(lora.weight, base.weight, atol=1e-6)
    x = torch.randn(5, 12, device="cuda")
    assert torch.allclose(lora(x), base(x), atol=1e-5)


def test_lora_linear_nonidentity_after_perturbation():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        lora.lora_B.add_(0.5)  # break the zero-init so the delta is nonzero
    x = torch.randn(3, 8)
    forward_out = lora(x)
    merged_out = torch.nn.functional.linear(x, lora.weight, lora.bias)
    # forward() and the merged-weight property must agree (dropout=0).
    assert torch.allclose(forward_out, merged_out, atol=1e-5)
    assert not torch.allclose(forward_out, base(x), atol=1e-4)


def test_lora_linear_only_factors_trainable():
    base = nn.Linear(6, 6)
    lora = LoRALinear(base, rank=2)
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert set(trainable) == {"lora_A", "lora_B"}


# --------------------------------------------------------------------------- #
# Patching a toy OpenCLIP-shaped visual tower
# --------------------------------------------------------------------------- #
def test_patch_toy_visual_tower():
    toy = ToyCLIP(dim=16, depth=3)
    info = apply_lora_to_openclip_visual(toy, rank=4, alpha=8.0, max_layers=2)
    # Last 2 of 3 blocks, each with out_proj + c_fc + c_proj => 6 modules.
    assert info["num_patched_modules"] == 6
    assert info["num_visual_blocks"] == 3
    assert info["patched_block_indices"] == [1, 2]
    assert all(name.startswith("visual.transformer.resblocks.") for name in info["patched_modules"])
    assert any("mlp.c_fc" in n for n in info["patched_modules"])
    assert any("mlp.c_proj" in n for n in info["patched_modules"])
    assert any("attn.out_proj" in n for n in info["patched_modules"])
    assert info["lora_used"] is True


def test_only_lora_trainable_after_patch():
    toy = ToyCLIP(dim=16, depth=3)
    apply_lora_to_openclip_visual(toy, rank=4, max_layers=2)
    trainable = [n for n, p in toy.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all("lora_A" in n or "lora_B" in n for n in trainable)
    # The frozen base weights are explicitly not trainable.
    assert not any(n.endswith(".base.weight") and p.requires_grad for n, p in toy.named_parameters())


def test_trainable_count_nonzero_and_small():
    toy = ToyCLIP(dim=16, depth=3)
    total = sum(p.numel() for p in toy.parameters())
    info = apply_lora_to_openclip_visual(toy, rank=4, max_layers=2)
    trainable = count_trainable_parameters(toy)
    assert trainable == info["trainable_param_count"]
    assert trainable > 0
    assert trainable < total  # a small fraction of the backbone
    assert len(lora_module_names(toy)) == 6


def test_zero_target_modules_raises():
    toy = ToyCLIP(dim=16, depth=3)
    with pytest.raises(ValueError, match="zero target modules"):
        apply_lora_to_openclip_visual(toy, target_modules=("does_not_exist",), max_layers=2)


def test_missing_visual_tower_raises():
    class NoVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="visual-tower"):
        apply_lora_to_openclip_visual(NoVisual())


# --------------------------------------------------------------------------- #
# Tiny manual-LoRA training on synthetic tensors
# --------------------------------------------------------------------------- #
def test_tiny_manual_lora_training_one_step():
    model, info = _toy_manual_lora_model()
    dataset = _tiny_image_dataset()
    cfg = _tiny_cfg()

    before = [p.detach().clone() for p in model.trainable_parameters()]
    train_manual_lora(model, dataset, "plain_ft", cfg)
    after = list(model.trainable_parameters())

    logits = model.image_logits(dataset.val.observed, grad=False)
    assert logits.shape == (len(dataset.val), dataset.num_classes)
    assert torch.isfinite(logits).all()
    # At least one LoRA factor moved (B started at zero, so it must update).
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_frozen_mode_is_identity():
    model, _ = _toy_manual_lora_model()
    dataset = _tiny_image_dataset()
    cfg = _tiny_cfg()
    train_manual_lora(model, dataset, "frozen", cfg)
    # After a "frozen" call the LoRA factors are reset to identity (B == 0).
    for module in model.model.modules():
        if isinstance(module, LoRALinear):
            assert torch.count_nonzero(module.lora_B) == 0


def test_run_manual_lora_modes_and_go_no_go():
    model, info = _toy_manual_lora_model()
    dataset = _tiny_image_dataset()
    cfg = _tiny_cfg()
    out = run_manual_lora_modes(model, dataset, cfg, info["trainable_param_count"])
    modes = out["modes"]
    assert set(modes) == {"frozen", "plain_ft", "cf_aug", "csa"}
    assert out["trainable_params"]["frozen"] == 0
    assert out["trainable_params"]["csa"] == info["trainable_param_count"] > 0
    for m in modes.values():
        for key in ("clean_accuracy", "shortcut_accuracy", "counterfactual_instability",
                    "held_out_shortcut_accuracy", "zero_shot_clean_accuracy"):
            assert key in m
    go = compute_manual_lora_go_no_go(modes, cfg)
    for key in ("instability_drop_rel_vs_plain", "clean_accuracy_drop_vs_frozen",
                "held_out_overlay_gain_vs_plain", "held_out_overlay_delta_vs_cf_aug",
                "transfer_delta_vs_plain", "manual_lora_promising", "manual_lora_strong"):
        assert key in go
    assert isinstance(go["manual_lora_promising"], bool)
    assert isinstance(go["manual_lora_strong"], bool)
