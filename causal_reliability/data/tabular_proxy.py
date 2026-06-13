from __future__ import annotations

import torch

from causal_reliability.data.splits import DatasetBundle, tensor_dataset


def _make_split(n: int, corr: float, noise: float = 0.55) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stable = torch.randn(n, 3)
    logits = stable[:, 0] + 0.7 * stable[:, 1] - 0.5 * stable[:, 2]
    y = (logits > 0).long()
    agree = torch.rand(n) < corr
    proxy = torch.where(agree, y, 1 - y)
    proxy_onehot = torch.nn.functional.one_hot(proxy, 2).float()
    observed = torch.cat([stable + noise * torch.randn_like(stable), proxy_onehot], dim=1)
    return observed, y, proxy, y.clone()


def make_tabular_task(
    n_train: int = 512,
    n_test: int = 256,
    train_corr: float = 0.95,
    id_corr: float = 0.95,
    shift_corr: float = 0.1,
) -> DatasetBundle:
    return DatasetBundle(
        tensor_dataset(*_make_split(n_train, train_corr)),
        tensor_dataset(*_make_split(n_test, id_corr)),
        tensor_dataset(*_make_split(n_test, shift_corr)),
        input_shape=(5,),
        task_type="tabular",
    )
