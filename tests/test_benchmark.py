import numpy as np
import pytest

from nanomaternalpfn.benchmark import (
    brier_score,
    expected_calibration_error,
)


def test_brier_score():
    y = np.array([0, 1])
    p = np.array([0.25, 0.75])

    assert brier_score(y, p) == pytest.approx(0.0625)


def test_ece_is_zero_for_matching_bin_rates():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.0)


def test_ece_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        expected_calibration_error(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            n_bins=0,
        )
