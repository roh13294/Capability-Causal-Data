from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model
