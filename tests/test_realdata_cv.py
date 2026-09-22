import numpy as np
import pytest

from nanomaternalpfn.realdata_cv import (
    FoldMetrics,
    sample_context_indices,
    strict_cv_splits,
    summarize_folds,
)


def test_strict_cv_splits_cover_each_row_once_without_overlap():
    y = np.array([0] * 80 + [1] * 20)
    splits = strict_cv_splits(y, seed=42)

    assert len(splits) == 5

    all_test = []
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        all_test.extend(test_idx.tolist())

    assert sorted(all_test) == list(range(len(y)))


def test_context_sampling_stays_inside_outer_train_fold():
    y = np.array([0] * 80 + [1] * 20)
    train_idx, test_idx = strict_cv_splits(y, seed=1)[0]

    context_idx = sample_context_indices(
        train_idx,
        y,
        n_context=40,
        seed=123,
    )

    assert len(context_idx) == 40
    assert set(context_idx).issubset(set(train_idx))
    assert set(context_idx).isdisjoint(set(test_idx))
    assert set(np.unique(y[context_idx])) == {0, 1}


def test_fold_summary_uses_five_fold_values():
    rows = []
    for fold, accuracy in enumerate(
        [0.70, 0.75, 0.80, 0.85, 0.90],
        start=1,
    ):
        for model in (
            "nanoMaternalPFN",
            "logistic_regression",
            "random_forest",
        ):
            rows.append(
                FoldMetrics(
                    fold=fold,
                    model=model,
                    accuracy=accuracy,
                    balanced_accuracy=accuracy,
                    auroc=accuracy,
                    log_loss=1.0 - accuracy,
                    brier=1.0 - accuracy,
                    ece=0.1,
                )
            )

    summaries = summarize_folds(rows)
    pfn = next(
        summary
        for summary in summaries
        if summary.model == "nanoMaternalPFN"
    )

    assert pfn.metrics["accuracy"].mean == pytest.approx(0.80)
    assert pfn.metrics["accuracy"].std > 0
    assert (
        pfn.metrics["accuracy"].ci95_low
        < pfn.metrics["accuracy"].mean
        < pfn.metrics["accuracy"].ci95_high
    )
