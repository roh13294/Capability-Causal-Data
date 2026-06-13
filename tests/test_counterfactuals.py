import torch

from causal_reliability.counterfactuals import make_counterfactual_batch
from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.data.tabular_proxy import make_tabular_task
from causal_reliability.data.text_shortcuts import make_text_task


def test_vector_counterfactuals_preserve_causal_feature_and_change_shortcut():
    x = make_vector_task(n_train=8, n_test=4).train.tensors[0]
    cf = make_counterfactual_batch(x, "vector", 4)
    assert cf.shape == (8, 4, 4)
    assert torch.allclose(cf[:, :, 0], x[:, 0].unsqueeze(1))
    assert cf[:, :, 1].std() > 0


def test_text_and_tabular_counterfactuals_change_shortcut_slots():
    text_x = make_text_task(n_train=8, n_test=4).train.tensors[0]
    text_cf = make_counterfactual_batch(text_x, "text", 4)
    assert (text_cf[:, :, 2] != text_x[:, 2].unsqueeze(1)).any()
    tab_x = make_tabular_task(n_train=8, n_test=4).train.tensors[0]
    tab_cf = make_counterfactual_batch(tab_x, "tabular", 2)
    assert torch.allclose(tab_cf[:, :, :3], tab_x[:, :3].unsqueeze(1).expand(-1, 2, -1))
