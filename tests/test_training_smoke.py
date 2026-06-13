import torch
from torch.utils.data import DataLoader

from causal_reliability.counterfactuals import make_counterfactual_batch
from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.models import build_model
from causal_reliability.training.eval import evaluate
from causal_reliability.training.loops import train_model


def test_training_smoke_runs_for_few_batches():
    bundle = make_vector_task(n_train=32, n_test=16)
    model = build_model(bundle.task_type, bundle.input_shape)
    loader = DataLoader(bundle.train, batch_size=16)
    losses = train_model(
        model,
        loader,
        torch.device("cpu"),
        epochs=1,
        mode="combined",
        make_counterfactuals=lambda x: make_counterfactual_batch(x, "vector", 2),
    )
    assert len(losses) == 1
    assert 0 <= evaluate(model, loader, torch.device("cpu"))["accuracy"] <= 1
