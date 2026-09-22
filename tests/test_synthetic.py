import numpy as np
import pytest

from nanomaternalpfn.synthetic import FEATURE_BOUNDS, FEATURE_NAMES, generate_task


def _combined(task):
    X = np.vstack([task.X_context, task.X_query])
    y = np.concatenate([task.y_context, task.y_query])
    return X, y


def test_default_shapes_and_types():
    task = generate_task(seed=42)

    assert task.X_context.shape == (100, 6)
    assert task.y_context.shape == (100,)
    assert task.X_query.shape == (50, 6)
    assert task.y_query.shape == (50,)
    assert task.X_context.dtype == np.float32
    assert task.y_context.dtype == np.int64
    assert task.feature_names == FEATURE_NAMES


def test_same_seed_is_reproducible():
    a = generate_task(seed=7)
    b = generate_task(seed=7)

    np.testing.assert_array_equal(a.X_context, b.X_context)
    np.testing.assert_array_equal(a.y_context, b.y_context)
    np.testing.assert_array_equal(a.X_query, b.X_query)
    np.testing.assert_array_equal(a.y_query, b.y_query)
    assert a.metadata == b.metadata


def test_different_seeds_generate_different_tasks():
    a = generate_task(seed=1)
    b = generate_task(seed=2)

    assert not np.array_equal(a.X_context, b.X_context)
    assert a.metadata != b.metadata


def test_values_are_finite_and_within_v0_bounds():
    task = generate_task(seed=11)
    X, _ = _combined(task)

    assert np.isfinite(X).all()

    for column, name in enumerate(FEATURE_NAMES):
        low, high = FEATURE_BOUNDS[name]
        assert np.all(X[:, column] >= low)
        assert np.all(X[:, column] <= high)


def test_binary_outcome_contains_both_classes():
    task = generate_task(seed=99)
    _, y = _combined(task)

    assert set(np.unique(y)) == {0, 1}


def test_custom_split():
    task = generate_task(seed=3, n_patients=64, n_context=40)

    assert task.X_context.shape == (40, 6)
    assert task.X_query.shape == (24, 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_patients": 3},
        {"n_patients": 10, "n_context": 1},
        {"n_patients": 10, "n_context": 9},
    ],
)
def test_invalid_sizes_raise(kwargs):
    with pytest.raises(ValueError):
        generate_task(seed=0, **kwargs)
