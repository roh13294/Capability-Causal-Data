import torch


def make_counterfactual_batch(x: torch.Tensor, task_type: str, n_counterfactuals: int = 4) -> torch.Tensor:
    makers = {
        "vector": vector_counterfactuals,
        "vision": vision_counterfactuals,
        "text": text_counterfactuals,
        "tabular": tabular_counterfactuals,
    }
    return makers[task_type](x, n_counterfactuals)


def vector_counterfactuals(x: torch.Tensor, n_counterfactuals: int = 4) -> torch.Tensor:
    out = x.unsqueeze(1).repeat(1, n_counterfactuals, 1)
    values = torch.linspace(-1.4, 1.4, n_counterfactuals, device=x.device)
    out[:, :, 1] = values.view(1, -1)
    return out


def _vision_object_mask(x: torch.Tensor) -> torch.Tensor:
    bg = x[:, :, :1, :1]
    return (x - bg).abs().mean(dim=1, keepdim=True) > 0.08


def _vision_texture(i: int, x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    if i % 3 == 0:
        pattern = ((xx // 2) % 2).float()
    elif i % 3 == 1:
        pattern = (((xx + yy) // 3) % 2).float()
    else:
        pattern = (((xx - yy).abs() // 2) % 2).float()
    return pattern.to(dtype=x.dtype).view(1, 1, h, w)


def vision_counterfactuals(x: torch.Tensor, n_counterfactuals: int = 4, intervention_type: str = "color") -> torch.Tensor:
    colors = torch.tensor(
        [[0.95, 0.20, 0.20], [0.15, 0.45, 0.95], [0.15, 0.85, 0.35], [0.90, 0.75, 0.15]],
        device=x.device,
        dtype=x.dtype,
    )
    colors = colors[:n_counterfactuals]
    if colors.shape[0] < n_counterfactuals:
        colors = colors.repeat((n_counterfactuals + 3) // 4, 1)[:n_counterfactuals]
    mask = _vision_object_mask(x)
    out = x.unsqueeze(1).repeat(1, n_counterfactuals, 1, 1, 1)
    for i in range(n_counterfactuals):
        color_img = colors[i].view(1, 3, 1, 1).expand(x.shape[0], -1, x.shape[-2], x.shape[-1])
        if intervention_type == "color":
            out[:, i] = torch.where(mask, color_img, out[:, i])
        elif intervention_type == "background":
            out[:, i] = torch.where(mask, out[:, i], color_img * 0.55)
        elif intervention_type == "texture":
            texture = 0.55 + 0.35 * _vision_texture(i, x)
            textured = out[:, i] * texture.expand_as(out[:, i])
            out[:, i] = torch.where(mask, textured, out[:, i])
        else:
            raise ValueError(f"unknown intervention_type: {intervention_type}")
    return out.clamp(0, 1)


def text_counterfactuals(x: torch.Tensor, n_counterfactuals: int = 4) -> torch.Tensor:
    tokens = torch.tensor([5, 6, 7, 8], device=x.device, dtype=x.dtype)
    tokens = tokens.repeat((n_counterfactuals + 3) // 4)[:n_counterfactuals]
    out = x.unsqueeze(1).repeat(1, n_counterfactuals, 1)
    out[:, :, 2] = tokens.view(1, -1)
    return out


def tabular_counterfactuals(x: torch.Tensor, n_counterfactuals: int = 4) -> torch.Tensor:
    out = x.unsqueeze(1).repeat(1, n_counterfactuals, 1)
    proxy_values = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=x.device, dtype=x.dtype)
    for i in range(n_counterfactuals):
        out[:, i, -2:] = proxy_values[i % 2]
    return out
