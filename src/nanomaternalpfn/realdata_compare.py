"""Paired fold-wise statistical comparisons for strict UCI evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .realdata_cv import (
    METRIC_NAMES,
    T_95_DF4,
    FoldMetrics,
    evaluate_strict_cv,
)


BASELINES = ("logistic_regression", "random_forest")


@dataclass(frozen=True)
class PairedDifference:
    baseline: str
    metric: str
    mean_difference: float
    std_difference: float
    ci95_low: float
    ci95_high: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci95_low > 0.0 or self.ci95_high < 0.0


def paired_fold_differences(
    fold_metrics: list[FoldMetrics],
) -> list[PairedDifference]:
    """Compute nanoMaternalPFN-minus-baseline paired fold differences."""
    results: list[PairedDifference] = []

    for baseline in BASELINES:
        for metric in METRIC_NAMES:
            differences: list[float] = []

            for fold in range(1, 6):
                pfn = next(
                    row
                    for row in fold_metrics
                    if row.fold == fold
                    and row.model == "nanoMaternalPFN"
                )
                other = next(
                    row
                    for row in fold_metrics
                    if row.fold == fold
                    and row.model == baseline
                )
                differences.append(
                    float(getattr(pfn, metric) - getattr(other, metric))
                )

            values = np.asarray(differences, dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half_width = T_95_DF4 * std / sqrt(5)

            results.append(
                PairedDifference(
                    baseline=baseline,
                    metric=metric,
                    mean_difference=mean,
                    std_difference=std,
                    ci95_low=mean - half_width,
                    ci95_high=mean + half_width,
                )
            )

    return results


def _direction_note(metric: str) -> str:
    if metric in {"accuracy", "balanced_accuracy", "auroc"}:
        return "positive favors PFN"
    return "negative favors PFN"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired fold comparisons for strict UCI frozen transfer."
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

    metadata, fold_metrics, _ = evaluate_strict_cv(
        checkpoint=args.checkpoint,
        contexts_per_fold=args.contexts_per_fold,
        n_context=args.context,
        seed=args.seed,
        rf_trees=args.rf_trees,
        ece_bins=args.ece_bins,
        device_name=args.device,
    )

    comparisons = paired_fold_differences(fold_metrics)

    print("UCI Maternal Health Risk — paired outer-fold comparisons")
    print("difference = nanoMaternalPFN - baseline")
    print(
        f"cleaned rows: {metadata['final_rows']} | "
        f"outer folds: {metadata['outer_folds']} | "
        f"contexts/fold: {metadata['contexts_per_fold']}"
    )
    print()

    for baseline in BASELINES:
        print(f"vs {baseline}")
        for result in comparisons:
            if result.baseline != baseline:
                continue
            flag = "yes" if result.excludes_zero else "no"
            print(
                f"  {result.metric:17s} "
                f"{result.mean_difference:+.4f} ± "
                f"{result.std_difference:.4f} "
                f"(95% CI {result.ci95_low:+.4f} to "
                f"{result.ci95_high:+.4f}) | "
                f"CI excludes 0: {flag} | "
                f"{_direction_note(result.metric)}"
            )
        print()


if __name__ == "__main__":
    main()
