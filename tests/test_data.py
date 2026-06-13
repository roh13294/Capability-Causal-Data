import torch

from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.data.tabular_proxy import make_tabular_task
from causal_reliability.data.text_shortcuts import make_text_task
from causal_reliability.data.colored_digits import make_colored_digits_task


def _corr(ds):
    y = ds.tensors[1]
    s = ds.tensors[2]
    return float((y == s).float().mean())


def test_vector_data_shapes_and_shift_correlations():
    bundle = make_vector_task(n_train=100, n_test=80, train_corr=0.9, id_corr=0.9, shift_corr=0.1)
    assert bundle.train.tensors[0].shape == (100, 4)
    assert _corr(bundle.train) > 0.75
    assert _corr(bundle.shifted_test) < 0.3


def test_all_task_factories_return_expected_shapes():
    assert make_shape_task(n_train=8, n_test=4).train.tensors[0].shape[1:] == torch.Size([3, 16, 16])
    assert make_colored_digits_task(n_train=8, n_test=4).train.tensors[0].shape[1:] == torch.Size([3, 8, 8])
    assert make_text_task(n_train=8, n_test=4).train.tensors[0].shape[1:] == torch.Size([6])
    assert make_tabular_task(n_train=8, n_test=4).train.tensors[0].shape[1:] == torch.Size([5])
