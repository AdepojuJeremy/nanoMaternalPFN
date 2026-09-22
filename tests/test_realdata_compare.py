import pytest

from nanomaternalpfn.realdata_compare import paired_fold_differences
from nanomaternalpfn.realdata_cv import FoldMetrics


def _row(fold, model, value):
    return FoldMetrics(
        fold=fold,
        model=model,
        accuracy=value,
        balanced_accuracy=value,
        auroc=value,
        log_loss=1.0 - value,
        brier=1.0 - value,
        ece=1.0 - value,
    )


def test_paired_fold_differences_preserve_pairing_and_sign():
    rows = []
    for fold in range(1, 6):
        rows.extend(
            [
                _row(fold, "nanoMaternalPFN", 0.80),
                _row(fold, "logistic_regression", 0.70),
                _row(fold, "random_forest", 0.85),
            ]
        )

    results = paired_fold_differences(rows)

    pfn_vs_logistic_accuracy = next(
        result
        for result in results
        if result.baseline == "logistic_regression"
        and result.metric == "accuracy"
    )
    pfn_vs_rf_accuracy = next(
        result
        for result in results
        if result.baseline == "random_forest"
        and result.metric == "accuracy"
    )
    pfn_vs_logistic_loss = next(
        result
        for result in results
        if result.baseline == "logistic_regression"
        and result.metric == "log_loss"
    )

    assert pfn_vs_logistic_accuracy.mean_difference == pytest.approx(0.10)
    assert pfn_vs_rf_accuracy.mean_difference == pytest.approx(-0.05)
    assert pfn_vs_logistic_loss.mean_difference == pytest.approx(-0.10)
    assert pfn_vs_logistic_accuracy.excludes_zero
