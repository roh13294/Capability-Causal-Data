"""Real-model validation helpers for controlled shortcut experiments."""

from typing import Any


def load_real_model(*args: Any, **kwargs: Any):
    from causal_reliability.real_models.pretrained_loader import load_real_model as _load_real_model

    return _load_real_model(*args, **kwargs)

__all__ = ["load_real_model"]
