import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for x, y, _shortcut, _causal in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return {"accuracy": correct / max(total, 1), "loss": float(sum(losses) / max(len(losses), 1))}


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_rows = []
    label_rows = []
    for x, y, _shortcut, _causal in loader:
        logits_rows.append(model(x.to(device)).cpu())
        label_rows.append(y.cpu())
    return torch.cat(logits_rows), torch.cat(label_rows)
