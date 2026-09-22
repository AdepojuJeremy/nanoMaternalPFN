import numpy as np
import pytest
import torch

from nanomaternalpfn import NanoMaternalPFN, count_parameters, generate_task


def _batch_from_tasks(seeds):
    tasks = [generate_task(seed=s) for s in seeds]
    x_context = torch.from_numpy(
        np.stack([task.X_context for task in tasks])
    )
    y_context = torch.from_numpy(
        np.stack([task.y_context for task in tasks])
    )
    x_query = torch.from_numpy(
        np.stack([task.X_query for task in tasks])
    )
    return x_context, y_context, x_query


def test_forward_shape():
    model = NanoMaternalPFN()
    x_context, y_context, x_query = _batch_from_tasks([0, 1])

    logits = model(x_context, y_context, x_query)

    assert logits.shape == (2, 50, 2)
    assert torch.isfinite(logits).all()


def test_backward_pass():
    model = NanoMaternalPFN()
    x_context, y_context, x_query = _batch_from_tasks([2, 3])

    logits = model(x_context, y_context, x_query)
    loss = logits.square().mean()
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_default_model_is_small():
    model = NanoMaternalPFN()

    assert count_parameters(model) < 1_000_000


def test_query_predictions_do_not_depend_on_other_query_rows():
    torch.manual_seed(0)
    model = NanoMaternalPFN()
    model.eval()

    x_context, y_context, x_query = _batch_from_tasks([4])

    with torch.no_grad():
        reference = model(x_context, y_context, x_query)

        changed_query = x_query.clone()
        changed_query[:, 1:] = changed_query[:, 1:] + 1000.0
        changed = model(x_context, y_context, changed_query)

    torch.testing.assert_close(reference[:, :1], changed[:, :1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 95, "n_heads": 4},
        {"n_layers": 0},
        {"n_outputs": 1},
    ],
)
def test_invalid_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        NanoMaternalPFN(**kwargs)
