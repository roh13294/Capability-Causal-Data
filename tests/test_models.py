from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.data.text_shortcuts import make_text_task
from causal_reliability.models import build_model


def test_model_forward_passes():
    for bundle in [make_vector_task(n_train=4, n_test=4), make_shape_task(n_train=4, n_test=4), make_text_task(n_train=4, n_test=4)]:
        model = build_model(bundle.task_type, bundle.input_shape)
        logits = model(bundle.train.tensors[0])
        assert logits.shape == (4, 2)
