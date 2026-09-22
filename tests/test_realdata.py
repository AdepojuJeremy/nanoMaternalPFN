import numpy as np
import pytest

from nanomaternalpfn.realdata import (
    clean_duplicate_rows,
    episode_indices,
    map_risk_labels,
)


def test_map_risk_labels():
    values = np.array(["low risk", "mid risk", "high risk"])

    mapped = map_risk_labels(values)

    np.testing.assert_array_equal(mapped, np.array([0, 0, 1]))


def test_map_risk_labels_rejects_unknown_values():
    with pytest.raises(ValueError):
        map_risk_labels(np.array(["low risk", "unknown"]))


def test_clean_duplicate_rows_removes_exact_and_ambiguous_rows():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1])

    cleaned_X, cleaned_y, stats = clean_duplicate_rows(X, y)

    np.testing.assert_array_equal(cleaned_X, np.array([[1, 2], [5, 6]]))
    np.testing.assert_array_equal(cleaned_y, np.array([0, 1]))
    assert stats["exact_duplicates_removed"] == 1
    assert stats["ambiguous_rows_removed"] == 2


def test_episode_indices_are_reproducible_disjoint_and_stratified():
    y = np.array([0] * 100 + [1] * 100)

    context_a, query_a = episode_indices(
        y,
        n_context=100,
        n_query=50,
        seed=42,
    )
    context_b, query_b = episode_indices(
        y,
        n_context=100,
        n_query=50,
        seed=42,
    )

    np.testing.assert_array_equal(context_a, context_b)
    np.testing.assert_array_equal(query_a, query_b)
    assert set(context_a).isdisjoint(set(query_a))
    assert y[context_a].mean() == pytest.approx(0.5)
    assert y[query_a].mean() == pytest.approx(0.5)
