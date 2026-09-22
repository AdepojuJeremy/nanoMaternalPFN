"""Strict frozen-transfer evaluation on UCI Maternal Health Risk.

Protocol:
- clean the dataset once
- create 5 stratified outer folds
- keep each outer test fold completely out of every context set
- sample repeated 100-row contexts only from the other 4 folds
- evaluate the frozen PFN and task-fitted baselines on the full held-out fold
- average repeats within each fold
- report mean, sample standard deviation, and t-based 95% CI across 5 folds
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .benchmark import brier_score, expected_calibration_error
from .baselines import _binary_log_loss
from .model import NanoMaternalPFN
from .realdata import (
    _fit_logistic,
    _fit_random_forest,
    load_uci_maternal_health,
)
from .train import choose_device


MODEL_NAMES = (
    "nanoMaternalPFN",
    "logistic_regression",
    "random_forest",
)

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "log_loss",
    "brier",
    "ece",
)

# Student-t critical value for a two-sided 95% CI with df=4.
T_95_DF4 = 2.7764451051977987


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    model: str
    accuracy: float
    balanced_accuracy: float
    auroc: float
    log_loss: float
    brier: float
    ece: float


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class ModelSummary:
    model: str
    metrics: dict[str, MetricSummary]


def strict_cv_splits(
    y: np.ndarray,
    *,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the five stratified outer train/test splits."""
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros((len(y), 1))
    return list(splitter.split(dummy, y))


def sample_context_indices(
    train_indices: np.ndarray,
    y: np.ndarray,
    *,
    n_context: int,
    seed: int,
) -> np.ndarray:
    """Sample one stratified context set strictly inside an outer train fold."""
    if n_context < 2:
        raise ValueError("n_context must be at least 2")
    if n_context >= len(train_indices):
        raise ValueError("n_context must be smaller than outer training pool")

    outer_y = y[train_indices]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_context,
        random_state=seed,
    )
    local_context, _ = next(
        splitter.split(np.zeros((len(train_indices), 1)), outer_y)
    )
    return train_indices[local_context]


def _metrics(y_true: np.ndarray, p1: np.ndarray, *, ece_bins: int) -> dict[str, float]:
    pred = (p1 >= 0.5).astype(np.int64)
    return {
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, p1)),
        "log_loss": _binary_log_loss(y_true, p1),
        "brier": brier_score(y_true, p1),
        "ece": expected_calibration_error(y_true, p1, n_bins=ece_bins),
    }


def _mean_repeat_metrics(
    values: list[dict[str, float]],
) -> dict[str, float]:
    return {
        metric: float(np.mean([value[metric] for value in values]))
        for metric in METRIC_NAMES
    }


def summarize_folds(
    fold_metrics: list[FoldMetrics],
) -> list[ModelSummary]:
    """Summarize fold-level metrics with t-based 95% CIs."""
    summaries: list[ModelSummary] = []

    for model_name in MODEL_NAMES:
        model_folds = [
            result for result in fold_metrics if result.model == model_name
        ]
        if len(model_folds) != 5:
            raise ValueError(
                f"expected 5 folds for {model_name}, got {len(model_folds)}"
            )

        metric_summaries: dict[str, MetricSummary] = {}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [getattr(result, metric) for result in model_folds],
                dtype=np.float64,
            )
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half_width = T_95_DF4 * std / sqrt(5)

            metric_summaries[metric] = MetricSummary(
                mean=mean,
                std=std,
                ci95_low=mean - half_width,
                ci95_high=mean + half_width,
            )

        summaries.append(
            ModelSummary(
                model=model_name,
                metrics=metric_summaries,
            )
        )

    return summaries


def evaluate_strict_cv(
    *,
    checkpoint: str,
    contexts_per_fold: int = 20,
    n_context: int = 100,
    seed: int = 0,
    rf_trees: int = 100,
    ece_bins: int = 10,
    device_name: str = "auto",
) -> tuple[dict, list[FoldMetrics], list[ModelSummary]]:
    """Run strict 5-fold frozen-transfer evaluation."""
    if contexts_per_fold < 1:
        raise ValueError("contexts_per_fold must be at least 1")

    data = load_uci_maternal_health()
    device = choose_device(device_name)

    pfn = NanoMaternalPFN().to(device)
    pfn.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    pfn.eval()

    results: list[FoldMetrics] = []

    with torch.no_grad():
        for fold, (train_idx, test_idx) in enumerate(
            strict_cv_splits(data.y, seed=seed),
            start=1,
        ):
            X_query = data.X[test_idx]
            y_query = data.y[test_idx]

            repeat_metrics = {
                model_name: [] for model_name in MODEL_NAMES
            }

            for repeat in range(contexts_per_fold):
                context_seed = seed + fold * 10_000 + repeat
                context_idx = sample_context_indices(
                    train_idx,
                    data.y,
                    n_context=n_context,
                    seed=context_seed,
                )

                X_context = data.X[context_idx]
                y_context = data.y[context_idx]

                logits = pfn(
                    torch.from_numpy(X_context).unsqueeze(0).to(device),
                    torch.from_numpy(y_context).unsqueeze(0).to(device),
                    torch.from_numpy(X_query).unsqueeze(0).to(device),
                )
                pfn_p1 = (
                    torch.softmax(logits, dim=-1)[0, :, 1]
                    .float()
                    .cpu()
                    .numpy()
                )

                logistic = _fit_logistic(X_context, y_context)
                logistic_p1 = logistic.predict_proba(X_query)[:, 1]

                forest = _fit_random_forest(
                    X_context,
                    y_context,
                    rf_trees,
                )
                forest_p1 = forest.predict_proba(X_query)[:, 1]

                predictions = {
                    "nanoMaternalPFN": pfn_p1,
                    "logistic_regression": logistic_p1,
                    "random_forest": forest_p1,
                }

                for model_name, p1 in predictions.items():
                    repeat_metrics[model_name].append(
                        _metrics(y_query, p1, ece_bins=ece_bins)
                    )

            for model_name in MODEL_NAMES:
                averaged = _mean_repeat_metrics(
                    repeat_metrics[model_name]
                )
                results.append(
                    FoldMetrics(
                        fold=fold,
                        model=model_name,
                        **averaged,
                    )
                )

    metadata = {
        **data.metadata,
        "outer_folds": 5,
        "contexts_per_fold": contexts_per_fold,
        "context_rows": n_context,
    }
    return metadata, results, summarize_folds(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict 5-fold frozen-transfer evaluation on UCI data."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contexts-per-fold", type=int, default=20)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    args = parser.parse_args()

    metadata, fold_results, summaries = evaluate_strict_cv(
        checkpoint=args.checkpoint,
        contexts_per_fold=args.contexts_per_fold,
        n_context=args.context,
        seed=args.seed,
        rf_trees=args.rf_trees,
        ece_bins=args.ece_bins,
        device_name=args.device,
    )

    print("UCI Maternal Health Risk — strict frozen transfer")
    print("target: high risk vs low/mid risk")
    print(f"data source: {metadata['data_source']}")
    print(f"cleaned rows: {metadata['final_rows']}")
    print(
        f"outer folds: 5 | contexts/fold: "
        f"{metadata['contexts_per_fold']} | "
        f"context rows: {metadata['context_rows']}"
    )
    print()

    for fold in range(1, 6):
        test_result = next(
            result
            for result in fold_results
            if result.fold == fold
            and result.model == "nanoMaternalPFN"
        )
        print(
            f"fold {fold}: PFN accuracy {test_result.accuracy:.3f} | "
            f"AUROC {test_result.auroc:.3f}"
        )

    print()
    for summary in summaries:
        print(summary.model)
        for metric in METRIC_NAMES:
            value = summary.metrics[metric]
            print(
                f"  {metric:17s} "
                f"{value.mean:.4f} ± {value.std:.4f} "
                f"(95% CI {value.ci95_low:.4f}–{value.ci95_high:.4f})"
            )
        print()


if __name__ == "__main__":
    main()
